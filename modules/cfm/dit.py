import math

import torch
import torch.nn as nn

from modules.attentions import FFN, ConditioningEncoder, MultiHeadAttention


# modified from https://github.com/sh-lee-prml/HierSpeechpp/blob/main/modules.py#L390
class DiTConVBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_channels,
        filter_channels,
        num_heads,
        kernel_size=3,
        p_dropout=0.1,
        utt_emb_dim=0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_channels, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(
            hidden_channels, hidden_channels, num_heads, p_dropout=p_dropout
        )
        self.norm2 = nn.LayerNorm(hidden_channels, elementwise_affine=False, eps=1e-6)
        self.cross_attn = ConditioningEncoder(
            hidden_channels=hidden_channels,
            n_heads=num_heads,
            dim_head=None,
            p_dropout=p_dropout,
            cond_emb_dim=192,
        )
        self.norm3 = nn.LayerNorm(hidden_channels, elementwise_affine=False, eps=1e-6)
        self.mlp = FFN(
            hidden_channels,
            hidden_channels,
            filter_channels,
            kernel_size,
            p_dropout=p_dropout,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(utt_emb_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 9 * hidden_channels, bias=True),
        )

    def forward(self, x, c, x_mask, cond, cond_mask):
        """
        Args:
            x : [batch_size, channel, time]
            c : [batch_size, channel]
            x_mask : [batch_size, 1, time]
        return the same shape as x
        """
        x = x * x_mask

        # attn mask
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = (
            self.adaLN_modulation(c).unsqueeze(2).chunk(9, dim=1)
        )  # shape: [batch_size, channel, 1]

        # self attention
        modulated_x = self.modulate(
            self.norm1(x.transpose(1, 2)).transpose(1, 2), shift_msa, scale_msa
        )
        x = (
            x
            + gate_msa
            * self.attn(
                modulated_x,
                c=modulated_x,
                attn_mask=attn_mask,
            )
            * x_mask
        )

        # cross attention
        modulated_cross_x = self.modulate(
            self.norm2(x.transpose(1, 2)).transpose(1, 2), shift_mca, scale_mca
        )
        x = (
            x
            + gate_mca
            * self.cross_attn(modulated_cross_x, x_mask, cond, cond_mask)
            * x_mask
        )

        x = x + gate_mlp * self.mlp(
            self.modulate(
                self.norm3(x.transpose(1, 2)).transpose(1, 2), shift_mlp, scale_mlp
            ),
            x_mask,
        )

        return x

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift


class DitWrapper(nn.Module):
    """add FiLM layer to condition time embedding to DiT"""

    def __init__(
        self,
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
        self.time_fusion = FiLMLayer(hidden_channels, time_channels)
        self.conv_layers = nn.ModuleList(
            [
                ConvNeXtBlock(hidden_channels, filter_channels, utt_emb_dim)
                for _ in range(conv_layers)
            ]
        )
        self.block = DiTConVBlock(
            hidden_channels,
            hidden_channels,
            num_heads,
            kernel_size,
            p_dropout,
            utt_emb_dim,
        )

    def forward(self, x, c, t, x_mask, cond, cond_mask):
        x = self.time_fusion(x, t) * x_mask
        for layer in self.conv_layers:
            x = layer(x, c, x_mask)
        x = self.block(x, c, x_mask, cond, cond_mask)
        return x


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer
    Reference: https://arxiv.org/abs/1709.07871
    """

    def __init__(self, in_channels, cond_channels):
        super(FiLMLayer, self).__init__()
        self.in_channels = in_channels
        self.film = nn.Conv1d(cond_channels, in_channels * 2, 1)

    def forward(self, x, c):
        gamma, beta = torch.chunk(self.film(c.unsqueeze(2)), chunks=2, dim=1)
        return gamma * x + beta


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, filter_channels, gin_channels):
        super().__init__()
        self.dwconv = nn.Conv1d(
            in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels
        )
        self.norm = StyleAdaptiveLayerNorm(in_channels, gin_channels)
        self.pwconv = nn.Sequential(
            nn.Linear(in_channels, filter_channels),
            nn.GELU(),
            nn.Linear(filter_channels, in_channels),
        )

    def forward(self, x, c, x_mask) -> torch.Tensor:
        residual = x
        x = self.dwconv(x) * x_mask
        x = self.norm(x.transpose(1, 2), c)
        x = self.pwconv(x).transpose(1, 2)
        x = residual + x
        return x * x_mask


class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, in_channels, cond_channels):
        """
        Style Adaptive Layer Normalization (SALN) module.

        Parameters:
        in_channels: The number of channels in the input feature maps.
        cond_channels: The number of channels in the conditioning input.
        """
        super(StyleAdaptiveLayerNorm, self).__init__()
        self.in_channels = in_channels

        self.saln = nn.Linear(cond_channels, in_channels * 2, 1)
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.saln.bias.data[: self.in_channels], 1)
        nn.init.constant_(self.saln.bias.data[self.in_channels :], 0)

    def forward(self, x, c):
        gamma, beta = torch.chunk(self.saln(c.unsqueeze(1)), chunks=2, dim=-1)
        return gamma * self.norm(x) + beta


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
            nn.SiLU(inplace=True),
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

        self.time_embeddings = SinusoidalPosEmb(hidden_channels)
        self.time_mlp = TimestepEmbedding(
            hidden_channels, hidden_channels, filter_channels
        )

        # in projection
        self.in_proj = nn.Conv1d(in_channels, hidden_channels, 1)

        self.blocks = nn.ModuleList(
            [
                DitWrapper(
                    hidden_channels=hidden_channels,
                    filter_channels=filter_channels,
                    num_heads=n_heads,
                    kernel_size=kernel_size,
                    p_dropout=dropout,
                    utt_emb_dim=utt_emb_dim,
                    conv_layers=3,
                    time_channels=hidden_channels,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_proj = nn.Conv1d(hidden_channels, out_channels, 1)

        self.initialize_weights()

    def initialize_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.block.adaLN_modulation[-1].bias, 0)

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

        x = self.in_proj(x) * mask

        for block in self.blocks:
            x = block(x, spks, t, mask, cond, cond_mask)

        output = self.final_proj(x * mask)

        return output * mask
