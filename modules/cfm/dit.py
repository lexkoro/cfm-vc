import math

import torch
import torch.nn as nn
from einops import rearrange

from modules.attentions import FFN, MultiHeadAttention
from modules.modules import ConditionalLayerNorm, LayerNorm


class ConditionalGroupNorm(nn.Module):
    def __init__(self, groups, normalized_shape, context_dim):
        super().__init__()
        self.norm = nn.GroupNorm(groups, normalized_shape, affine=False)
        self.context_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(context_dim, 2 * normalized_shape)
        )
        self.context_mlp[1].weight.data.zero_()
        self.context_mlp[1].bias.data.zero_()

    def forward(self, x, context):
        context = self.context_mlp(context)
        ndims = " 1" * len(x.shape[2:])
        context = rearrange(context, f"b c -> b c{ndims}")

        scale, shift = context.chunk(2, dim=1)
        x = self.norm(x) * (scale + 1.0) + shift
        return x


class Block1D(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8, context_dim=None):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(dim, dim_out, 3, padding=1)
        if context_dim is None:
            self.norm = torch.nn.GroupNorm(groups, dim_out)
        else:
            self.norm = ConditionalGroupNorm(groups, dim_out, context_dim)
        self.mish = nn.Mish()

    def forward(self, x, mask, utt_emb=None):
        output = self.conv1d(x * mask)
        if utt_emb is not None:
            output = self.norm(output, utt_emb)
        else:
            output = self.norm(output)
        output = self.mish(output)
        return output * mask


class ResnetBlock1D(torch.nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8, context_dim=512):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out)
        )

        self.block1 = Block1D(dim, dim_out, groups=groups, context_dim=context_dim)
        self.block2 = Block1D(dim_out, dim_out, groups=groups, context_dim=context_dim)

        self.res_conv = (
            torch.nn.Conv1d(dim, dim_out, 1) if dim != dim_out else torch.nn.Identity()
        )

    def forward(self, x, mask, time_emb, utt_emb=None):
        h = self.block1(x, mask, utt_emb)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask, utt_emb)
        output = h + self.res_conv(x * mask)
        return output


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        time_channels,
        n_heads,
        n_layers,
        dim_head=None,
        kernel_size=1,
        p_dropout=0.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size

        self.attn_layers = nn.ModuleList()

        self.norm_layers_1 = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        self.ffn_layers_1 = nn.ModuleList()

        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    dim_head=dim_head,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_1.append(
                ConditionalLayerNorm(hidden_channels, time_channels)
            )

            self.ffn_layers_1.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size=kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(
                ConditionalLayerNorm(hidden_channels, time_channels)
            )

        self.norm = LayerNorm(hidden_channels)

    def forward(self, x, x_mask, t):
        # attn mask
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask

        for i in range(self.n_layers):
            # self-attention
            attn_input = self.norm_layers_1[i](x, t)
            x = self.attn_layers[i](attn_input, attn_input, attn_mask) + x

            # feed-forward
            ffn_input = self.norm_layers_2[i](x, t)
            x = self.ffn_layers_1[i](ffn_input, x_mask) + x

        return self.norm(x * x_mask)


class DitWrapper(nn.Module):
    """add FiLM layer to condition time embedding to DiT"""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        filter_channels,
        num_heads,
        kernel_size=3,
        p_dropout=0.1,
        utt_emb_dim=0,
        conv_layers=3,
        time_channels=0,
    ):
        super().__init__()

        self.conv_layers = nn.ModuleList([])

        for _ in range(conv_layers):
            self.conv_layers.append(
                ResnetBlock1D(
                    dim=in_channels,
                    dim_out=hidden_channels,
                    time_emb_dim=time_channels,
                    groups=8,
                    context_dim=utt_emb_dim,
                )
            )
            in_channels = hidden_channels

        self.block = Encoder(
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            time_channels=time_channels,
            n_heads=num_heads,
            n_layers=1,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
        )

    def forward(self, x, c, t, x_mask, cond, cond_mask):
        for layer in self.conv_layers:
            x = layer(x, x_mask, t, c)
        x = self.block(x, x_mask, t)
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
    def __init__(self, in_channels, out_channels, filter_channels):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_channels, filter_channels),
            nn.SiLU(),
            nn.Linear(filter_channels, out_channels),
        )

    def forward(self, x):
        return self.layer(x)


# reference: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/decoder.py
class DiT(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        filter_channels,
        dropout=0.05,
        n_layers=1,
        n_heads=4,
        kernel_size=3,
        utt_emb_dim=0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.n_layers = n_layers

        self.time_embeddings = SinusoidalPosEmb(hidden_channels)
        self.time_mlp = TimestepEmbedding(
            hidden_channels, hidden_channels, filter_channels
        )

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for idx in range(n_layers // 2):
            self.down_blocks.append(
                DitWrapper(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    filter_channels=filter_channels,
                    num_heads=n_heads,
                    kernel_size=kernel_size,
                    p_dropout=dropout,
                    utt_emb_dim=utt_emb_dim,
                    conv_layers=2,
                    time_channels=hidden_channels,
                )
            )
            in_channels = hidden_channels

        for idx in range(n_layers // 2):
            self.up_blocks.append(
                DitWrapper(
                    in_channels=hidden_channels * 2,
                    hidden_channels=hidden_channels,
                    filter_channels=filter_channels,
                    num_heads=n_heads,
                    kernel_size=kernel_size,
                    p_dropout=dropout,
                    utt_emb_dim=utt_emb_dim,
                    conv_layers=2,
                    time_channels=hidden_channels,
                )
            )

        self.final_block = Block1D(hidden_channels, hidden_channels)
        self.final_proj = nn.Conv1d(hidden_channels, out_channels, 1)

    #     self.initialize_weights()

    # def initialize_weights(self):
    #     for block in self.blocks:
    #         nn.init.constant_(block.block.adaLN_modulation[-1].weight, 0)
    #         nn.init.constant_(block.block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, mask, mu, t, spks=None, cond=None, cond_mask=None):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            c (_type_): shape (batch_size, gin_channels)

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        t = self.time_mlp(self.time_embeddings(t))
        x = torch.cat((x, mu), dim=1)

        skip_connections = []
        for idx, block in enumerate(self.down_blocks):
            x = block(x, spks, t, mask, cond, cond_mask)
            skip_connections.append(x)

        for idx, block in enumerate(self.up_blocks):
            skip_x = skip_connections.pop()
            x = torch.cat([x, skip_x], dim=1)
            x = block(x, spks, t, mask, cond, cond_mask)

        x = self.final_block(x, mask)
        output = self.final_proj(x * mask)

        return output * mask
