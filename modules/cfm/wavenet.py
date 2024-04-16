# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

"""WaveNet modules.

This code is modified from https://github.com/kan-bayashi/ParallelWaveGAN.

"""


class Block1D(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock1D(torch.nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out)
        )

        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)

        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(
        self,
        channels,
        use_conv=False,
        use_conv_transpose=True,
        out_channels=None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs):
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)

        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs


class Downsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class FiLM(nn.Module):
    def __init__(self, in_dim, cond_dim):
        super().__init__()

        self.gain = nn.Linear(cond_dim, in_dim)
        self.bias = nn.Linear(cond_dim, in_dim)

        nn.init.xavier_uniform_(self.gain.weight)
        nn.init.constant_(self.gain.bias, 1)

        nn.init.xavier_uniform_(self.bias.weight)
        nn.init.constant_(self.bias.bias, 0)

    def forward(self, x, condition):
        gain = self.gain(condition)
        bias = self.bias(condition)
        if gain.dim() == 2:
            gain = gain.unsqueeze(-1)
        if bias.dim() == 2:
            bias = bias.unsqueeze(-1)
        return x * gain + bias


class ConditionalLayerNorm(nn.Module):
    """
    Conditional Layer Normalization module

    Args:
        embedding_dim (int): Dimension of the target embeddings.
        utt_emb_dim (int): Dimension of the speaker embeddings.
    """

    def __init__(self, embedding_dim: int, utt_emb_dim: int, epsilon: float = 1e-6):
        super(ConditionalLayerNorm, self).__init__()

        self.embedding_dim = embedding_dim
        self.utt_emb_dim = utt_emb_dim
        self.epsilon = epsilon
        self.W_scale = nn.Linear(self.utt_emb_dim, self.embedding_dim)
        self.W_bias = nn.Linear(self.utt_emb_dim, self.embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)

    def forward(self, x: torch.Tensor, utt_emb: torch.Tensor):
        x = x.transpose(1, 2)
        if utt_emb.dim() == 3:
            utt_emb = utt_emb.squeeze(-1)
        scale = self.W_scale(utt_emb).unsqueeze(1)
        bias = self.W_bias(utt_emb).unsqueeze(1)
        x = nn.functional.layer_norm(x, (self.embedding_dim,), eps=self.epsilon)
        x = x * scale + bias
        x = x.transpose(1, 2)
        return x


class Conv1d(torch.nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class Conv1d1x1(Conv1d):
    """1x1 Conv1d with customized initialization."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool):
        """Initialize 1x1 Conv1d module."""
        super().__init__(
            in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias
        )


class LearnedSinusoidalPosEmb(nn.Module):
    """used by @crowsonkb"""

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return fouriered


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = nn.SiLU()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = nn.SiLU()

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class ResidualBlock(torch.nn.Module):
    """Residual block module in WaveNet."""

    def __init__(
        self,
        kernel_size: int = 3,
        residual_channels: int = 64,
        gate_channels: int = 128,
        skip_channels: int = 64,
        time_channels: int = 64,
        cond_channels: int = 512,
        gin_channels: int = -1,
        dropout_rate: float = 0.0,
        dilation: int = 1,
        bias: bool = True,
        scale_residual: bool = False,
        use_cross_attn: bool = False,
    ):
        """Initialize ResidualBlock module.

        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            residual_channels (int): Number of channels for residual connection.
            skip_channels (int): Number of channels for skip connection.
            aux_channels (int): Number of local conditioning channels.
            dropout (float): Dropout probability.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            scale_residual (bool): Whether to scale the residual outputs.

        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.scale_residual = scale_residual
        self.use_cross_attn = use_cross_attn

        # check
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        assert gate_channels % 2 == 0

        # dilation conv
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = Conv1d(
            residual_channels,
            gate_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        # time conditioning
        self.time_proj = nn.Linear(time_channels, residual_channels)

        # global conditioning
        self.spk_cond = ConditionalLayerNorm(gate_channels, gin_channels)

        if self.use_cross_attn:
            self.attn = nn.MultiheadAttention(
                residual_channels, 4, 0.1, batch_first=True
            )
            self.film = FiLM(gate_channels, residual_channels)

            self.ln = ConditionalLayerNorm(residual_channels, gin_channels)

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2

        self.conv1x1_out = Conv1d1x1(
            gate_out_channels, residual_channels + skip_channels, bias=bias
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
        g_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, residual_channels, T).
            x_mask Optional[torch.Tensor]: Mask tensor (B, 1, T).
            c (Optional[Tensor]): Local conditioning tensor (B, aux_channels, T).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, 1).

        Returns:
            Tensor: Output tensor for residual connection (B, residual_channels, T).
            Tensor: Output tensor for skip connection (B, skip_channels, T).

        """
        residual = x
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # add time conditioning
        t = self.time_proj(t).unsqueeze(-1)
        x = x + t

        if x_mask is not None:
            x = x * x_mask

        if self.use_cross_attn:
            y_ = self.ln(x, g).transpose(1, 2)
            y_, _ = self.attn(y_, g_latent, g_latent)  # (B, T, d)

        # dilated conv
        x = self.conv(x)

        # spk cond layer norm
        x = self.spk_cond(x, g)

        if self.use_cross_attn:
            x = self.film(x.transpose(1, 2), y_)  # (B, T, 2*d)
            x = x.transpose(1, 2)  # (B, 2*d, T)

        if x_mask is not None:
            x = x * x_mask

        # split into two part for gated activation
        xa, xb = torch.chunk(x, 2, dim=1)

        x = torch.tanh(xa) * torch.sigmoid(xb)

        # residual + skip 1x1 conv
        x = self.conv1x1_out(x)
        if x_mask is not None:
            x = x * x_mask

        # split integrated conv results
        x, s = x.split([self.residual_channels, self.skip_channels], dim=1)

        # for residual connection
        x = x + residual
        if self.scale_residual:
            x = x * math.sqrt(0.5)

        return x, s


class WaveNet(torch.nn.Module):
    """WaveNet with global conditioning."""

    def __init__(
        self,
        kernel_size: int = 3,
        layers: int = 30,
        stacks: int = 3,
        cross_attn_per_layer: int = 3,
        base_dilation: int = 2,
        input_channels: int = 80,
        output_channels: int = 80,
        residual_channels: int = 64,
        gate_channels: int = 128,
        skip_channels: int = 64,
        global_channels: int = -1,
        dropout_rate: float = 0.0,
        bias: bool = True,
        use_weight_norm: bool = True,
        scale_residual: bool = False,
        scale_skip_connect: bool = False,
    ):
        """Initialize WaveNet module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of dilated convolution.
            layers (int): Number of residual block layers.
            stacks (int): Number of stacks i.e., dilation cycles.
            base_dilation (int): Base dilation factor.
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            global_channels (int): Number of channels for global conditioning feature.
            dropout_rate (float): Dropout rate. 0.0 means no dropout applied.
            bias (bool): Whether to use bias parameter in conv layer.
            use_weight_norm (bool): Whether to use weight norm. If set to true, it will
                be applied to all of the conv layers.
            scale_residual (bool): Whether to scale the residual outputs.
            scale_skip_connect (bool): Whether to scale the skip connection outputs.

        """
        super().__init__()
        self.input_channels = input_channels
        self.residual_channels = residual_channels
        self.kernel_size = kernel_size
        self.base_dilation = base_dilation
        self.scale_skip_connect = scale_skip_connect
        self.cross_attn_per_layer = cross_attn_per_layer
        self.layers_per_residual_block = 3

        # check the number of layers and stacks
        assert layers % stacks == 0

        # time position embedding
        self.time_embeddings = SinusoidalPosEmb(residual_channels)
        time_embed_dim = residual_channels * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=residual_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

        # first and last conv layers
        self.first_conv = Conv1d1x1(input_channels * 2, residual_channels, bias=True)
        self.final_block = Block1D(residual_channels, residual_channels)
        self.final_proj = nn.Conv1d(residual_channels, output_channels, 1)

        # mu cond layer norm
        self.cond_ln = ConditionalLayerNorm(residual_channels, global_channels)

        # define residual blocks
        self.first_residual = torch.nn.ModuleList([])
        self.down_blocks = nn.ModuleList([])
        for i in range(3):
            is_last = i == 3 - 1
            for layer in range(self.layers_per_residual_block):
                dilation = base_dilation ** (layer % stacks)
                self.first_residual.append(
                    ResidualBlock(
                        kernel_size=kernel_size,
                        residual_channels=residual_channels,
                        gate_channels=gate_channels,
                        skip_channels=skip_channels,
                        time_channels=time_embed_dim,
                        cond_channels=input_channels,
                        gin_channels=global_channels,
                        dilation=dilation,
                        dropout_rate=dropout_rate,
                        bias=bias,
                        scale_residual=scale_residual,
                        use_cross_attn=(layer % self.cross_attn_per_layer == 0),
                    )
                )
            downsample = (
                Downsample1D(residual_channels)
                if not is_last
                else nn.Conv1d(residual_channels, residual_channels, 3, padding=1)
            )
            self.down_blocks.append(nn.ModuleList([self.first_residual, downsample]))

        # define residual blocks
        self.mid_residual = torch.nn.ModuleList()
        for layer in range(6):
            dilation = base_dilation ** (layer % stacks)
            self.mid_residual.append(
                ResidualBlock(
                    kernel_size=kernel_size,
                    residual_channels=residual_channels,
                    gate_channels=gate_channels,
                    skip_channels=skip_channels,
                    time_channels=time_embed_dim,
                    cond_channels=input_channels,
                    gin_channels=global_channels,
                    dilation=dilation,
                    dropout_rate=dropout_rate,
                    bias=bias,
                    scale_residual=scale_residual,
                    use_cross_attn=(layer % self.cross_attn_per_layer == 0),
                )
            )

        # define residual blocks
        self.last_residual = torch.nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        for _ in range(2):
            for layer in range(self.layers_per_residual_block):
                dilation = base_dilation ** (layer % stacks)

                self.last_residual.append(
                    ResidualBlock(
                        kernel_size=kernel_size,
                        residual_channels=residual_channels,
                        gate_channels=gate_channels,
                        skip_channels=skip_channels,
                        time_channels=time_embed_dim,
                        cond_channels=input_channels,
                        gin_channels=global_channels,
                        dilation=dilation,
                        dropout_rate=dropout_rate,
                        bias=bias,
                        scale_residual=scale_residual,
                        use_cross_attn=(layer % self.cross_attn_per_layer == 0),
                    )
                )
            resnet = ResnetBlock1D(
                dim=2 * residual_channels,
                dim_out=residual_channels,
                time_emb_dim=time_embed_dim,
            )
            upsample = Upsample1D(residual_channels, use_conv_transpose=True)
            self.up_blocks.append(nn.ModuleList([resnet, self.last_residual, upsample]))

        self.layers = 21

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, mask, mu, t, spks=None, cond=None) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T) if use_first_conv else
                (B, residual_channels, T).
            x_mask (Optional[Tensor]): Mask tensor (B, 1, T).
            c (Optional[Tensor]): Local conditioning features (B, aux_channels, T).
            g (Optional[Tensor]): Global conditioning features (B, global_channels, 1).

        Returns:
            Tensor: Output tensor (B, out_channels, T) if use_last_conv else
                (B, residual_channels, T).
        """

        # time position embeddingd
        t = self.time_embeddings(t)
        t = self.time_mlp(t)

        # first conv layer
        x = torch.cat([x, mu], dim=1)
        x = self.first_conv(x) * mask

        # x cond layer norm
        x = self.cond_ln(x, spks)

        hiddens = []
        masks = [mask]
        # down residual block
        for residual, downsample in self.down_blocks:
            mask_down = masks[-1]

            # residual blocks
            skips = 0.0
            for layer in residual:
                x, h = layer(x, x_mask=mask_down, t=t, g=spks, g_latent=cond)
                skips = skips + h
            x = skips

            # downsample
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        # mid residual blocks
        skips = 0.0
        for layer in self.mid_residual:
            x, h = layer(x, x_mask=mask_mid, t=t, g=spks, g_latent=cond)
            skips = skips + h
        x = skips

        # up residual block
        for resnet, residual, upsample in self.up_blocks:
            mask_up = masks.pop()

            # resnet
            # concat hiddens and x
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet(x, mask_up, t)

            # residual blocks
            skips = 0.0
            for layer in residual:
                x, h = layer(x, x_mask=mask_up, t=t, g=spks, g_latent=cond)
                skips = skips + h
            x = skips

            # upsample
            x = upsample(x * mask_up)

        # last conv layers
        x = self.final_block(x, mask)
        output = self.final_proj(x * mask)

        return output * mask

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m: torch.nn.Module):
            try:
                remove_parametrizations(m, "weight")
                # logging.debug(f"Weight norm is removed from {m}.")
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                weight_norm(m)
                # logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)
