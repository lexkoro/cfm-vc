import math
from typing import Optional

import torch
import torch.nn as nn
from einops import repeat

from modules.attention.ffn import FFN
from modules.attention.multihead import MultiHeadAttention
from modules.modules import AdaLNZero, DepthwiseConv, get_normalization


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer
    Reference: https://arxiv.org/abs/1709.07871
    """

    def __init__(self, in_channels, time_channels):
        super(FiLMLayer, self).__init__()
        self.in_channels = in_channels
        self.film = nn.Conv1d(time_channels, in_channels * 2, 1)

    def forward(self, x, c):
        gamma, beta = torch.chunk(self.film(c), chunks=2, dim=1)
        return gamma * x + beta


class Block1D(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            dim, dim_out, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.norm = get_normalization("adarmsnorm", dim_out, time_emb_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, t):
        output = self.conv1d(x * mask)
        output = self.norm(output, t)
        output = self.act(output)
        output = self.dropout(output)
        return output * mask


class ResnetBlock1D(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        time_emb_dim: int = 512,
    ):
        super().__init__()

        self.block1 = Block1D(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            dropout=dropout,
            time_emb_dim=time_emb_dim,
        )
        self.block2 = Block1D(
            dim_out,
            dim_out,
            kernel_size=kernel_size,
            dropout=dropout,
            time_emb_dim=time_emb_dim,
        )
        self.res_conv = (
            torch.nn.Conv1d(dim_in, dim_out, 1)
            if dim_in != dim_out
            else torch.nn.Identity()
        )

    def forward(self, x, mask, t):
        h = self.block1(x, mask, t)
        h = self.block2(h, mask, t)
        output = h + self.res_conv(x * mask)
        return output


class Conv1dFourierEmbed(nn.Module):
    def __init__(
        self,
        channels,
        p=0.5,
    ):
        super().__init__()
        assert p <= 1.0

        dim_fourier = int(p * channels)
        dim_rest = channels - (dim_fourier * 2)

        if dim_rest < 0:
            raise ValueError(f"p={p} is too large for channels={channels}.")

        # Use Conv1d instead of Linear
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=dim_fourier + dim_rest,
            kernel_size=1,
            bias=False,
        )
        self.split_dims = (dim_fourier, dim_rest)

    def forward(self, x):
        # x is expected to be (batch, channels, time)
        hiddens = self.conv(x)
        # Split along the channel dimension (dim=1)
        fourier, rest = hiddens.split(self.split_dims, dim=1)
        # Concatenate along the channel dimension (dim=1)
        return torch.cat((fourier.sin(), fourier.cos(), rest), dim=1)


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        dim_head: Optional[int] = None,
        p_dropout: float = 0.0,
        cond_emb_dim: int = 0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads

        # self attention
        self.norm_1 = get_normalization("adarmsnorm", hidden_channels, cond_emb_dim)
        self.attn = MultiHeadAttention(
            channels=hidden_channels,
            out_channels=hidden_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            p_dropout=p_dropout,
        )
        self.norm_1_cond = AdaLNZero(hidden_channels, cond_emb_dim)

        # feed forward
        self.norm_2 = get_normalization("adarmsnorm", hidden_channels, cond_emb_dim)
        self.ffn = FFN(
            hidden_channels,
            hidden_channels,
            filter_channels,
            p_dropout=p_dropout,
            use_glu=True,
        )
        self.norm_2_cond = AdaLNZero(hidden_channels, cond_emb_dim)

    def forward(self, x, x_mask, attn_mask, t):
        # norm packed shape
        attn_in = self.norm_1(x, t)
        attn_out = self.attn(
            attn_in,
            c=attn_in,
            attn_mask=attn_mask,
        )
        x = self.norm_1_cond(attn_out, t) + x

        # feed-forward
        ff_input = self.norm_2(x, t)
        ff_out = self.ffn(ff_input, x_mask)
        x = self.norm_2_cond(ff_out, t) + x

        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.pos_emb = SinusoidalPosEmb(hidden_channels)
        self.layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.SiLU(),
            nn.Linear(hidden_channels * 2, hidden_channels),
        )

    def forward(self, t):
        t = self.pos_emb(t)
        t = self.layer(t)
        return t


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x, mask):
        x = self.conv1d(x) * mask

        return x


class DitWrapper(nn.Module):
    """add FiLM layer to condition time embedding to DiT"""

    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        dim_head=None,
        p_dropout=0.1,
        kernel_size=3,
        time_channels=0,
    ):
        super().__init__()

        self.depth = DepthwiseConv(dim=hidden_channels, kernel_size=kernel_size)
        self.block = Encoder(
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            p_dropout=p_dropout,
            cond_emb_dim=time_channels,
        )

    def forward(self, x, x_mask, attn_mask, t):
        x = self.depth(x, x_mask) + x
        x = self.block(x, x_mask, attn_mask, t)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        filter_channels: int,
        n_layers: int = 1,
        n_heads: int = 4,
        dim_head=None,
        kernel_size: int = 3,
        p_dropout: float = 0.05,
        time_channels: int = 512,
        use_skip_connections: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.n_layers = n_layers
        self.use_skip_connections = use_skip_connections

        # time embeddings
        self.time_mlp = TimestepEmbedding(time_channels)

        # prenet
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)

        # blocks
        self.blocks = nn.ModuleList()
        for idx in range(n_layers):
            is_last_half = idx >= n_layers // 2

            self.blocks.append(
                nn.ModuleList(
                    [
                        nn.Conv1d(hidden_channels * 2, hidden_channels, 1, bias=False)
                        if is_last_half and self.use_skip_connections
                        else nn.Identity(),
                        DitWrapper(
                            hidden_channels=hidden_channels,
                            filter_channels=filter_channels,
                            n_heads=n_heads,
                            dim_head=dim_head,
                            p_dropout=p_dropout,
                            kernel_size=kernel_size,
                            time_channels=time_channels,
                        ),
                    ]
                )
            )

        self.final_norm = get_normalization(
            "adarmsnorm", hidden_channels, time_channels
        )
        self.final_proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, mask, mu, cond_mel, t):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (torch.Tensor): shape (batch_size, 1, time)
            mu (torch.Tensor): shape (batch_size, inter_channels, time)
            cond_mel (torch.Tensor): shape (batch_size, cond_channels, time)
            t (torch.Tensor): shape (batch_size)
        Returns:
            (torch.Tensor): shape (batch_size, out_channels, time)


        """
        batch = x.shape[0]

        if t.ndim == 0:
            t = repeat(t, " -> b", b=batch)

        # time embeddings (style latents handled via cross-attn)
        t = self.time_mlp(t)

        x = torch.cat([x, cond_mel, mu], dim=1) * mask
        x = self.pre(x)

        # attn mask
        attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)

        # skip connections and speaker memory
        skips = []

        for idx, (skip_conv, block) in enumerate(self.blocks):  # type: ignore
            is_last_half = idx >= self.n_layers // 2

            # skip connection
            if self.use_skip_connections:
                if not is_last_half:
                    skips.append(x)
                else:
                    x = torch.cat([x, skips.pop()], dim=1)

            # skip
            x = skip_conv(x) * mask

            # encoder block
            x = block(x, mask, attn_mask, t)

        # final norm
        x = self.final_norm(x, t)
        output = self.final_proj(x * mask)

        return output
