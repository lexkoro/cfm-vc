from typing import Literal

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

NORM_TYPES = Literal[
    "layernorm",
    "rmsnorm",
    "adarmsnorm",
    "groupnorm",
    "condlayernorm",
    "condgroupnorm",
    "adainnorm",
]


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


def get_normalization(
    norm_type: NORM_TYPES = "rmsnorm",
    dim_out: int = 192,
    utt_emb_dim: int = 0,
):
    """
    Returns the appropriate normalization layer based on the norm_type and utt_emb_dim.

    Args:
        norm_type (str): The type of normalization to be applied.
        dim_out (int): The output dimension of the block.
        utt_emb_dim (int): The dimension of the utterance embedding.

    Returns:
        nn.Module: The normalization layer.
    """
    if norm_type == "condlayernorm" and utt_emb_dim != 0:
        return ConditionalLayerNorm(dim_out, utt_emb_dim)
    elif norm_type == "condgroupnorm" and utt_emb_dim != 0:
        return ConditionalGroupNorm(8, dim_out, utt_emb_dim)
    elif norm_type == "adainnorm" and utt_emb_dim != 0:
        return AdaIN1d(dim_out, utt_emb_dim)
    elif norm_type == "groupnorm":
        return nn.GroupNorm(8, dim_out)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim_out)
    elif norm_type == "adarmsnorm":
        return AdaptiveRMSNorm(dim_out, utt_emb_dim)
    elif norm_type == "layernorm":
        return LayerNorm(dim_out)
    else:
        raise NotImplementedError(
            f"Normalization type '{norm_type}' not implemented or utt_emb_dim is invalid."
        )


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (kernel_size,) = self.kernel_size
        (dilation,) = self.dilation
        (stride,) = self.stride

        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        causal_padded_x = F.pad(x, (self.causal_padding, 0), value=0.0)
        return super().forward(causal_padded_x)


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
    """
    Block1D module represents a 1-dimensional block in a neural network.

    Args:
        dim (int): The input dimension of the block.
        dim_out (int): The output dimension of the block.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        utt_emb_dim (int, optional): The dimension of the utterance embedding. Defaults to 0.
        norm_type (str, optional): The type of normalization to be applied.
            Possible values are "condlayernorm", "condgroupnorm" and "layernorm". Defaults to "layernorm".
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        utt_emb_dim: int = 0,
        norm_type: NORM_TYPES = "rmsnorm",
    ):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            dim, dim_out, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.norm_type = norm_type

        # get normalization layer
        self.norm = get_normalization(norm_type, dim_out, utt_emb_dim)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, utt_emb=None):
        """
        Forward pass of the Block1D module.

        Args:
            x (torch.Tensor): The input tensor.
            mask (torch.Tensor): The mask tensor.
            utt_emb (torch.Tensor, optional): The utterance embedding tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """

        output = self.conv1d(x * mask)
        if utt_emb is not None:
            output = self.norm(output, utt_emb)
        else:
            output = self.norm(output)
        output = self.act(output)
        output = self.dropout(output)
        return output * mask


class ResnetBlock1D(torch.nn.Module):
    """
    Residual block for 1D ResNet.

    Args:
        dim_in (int): Number of input channels.
        dim_out (int): Number of output channels.
        dropout (float, optional): Dropout rate. Default is 0.1.
        utt_emb_dim (int, optional): Dimension of utterance embedding. Default is 0.
        norm_type (str, optional): The type of normalization to be applied.
            Possible values are "condlayernorm", "condgroupnorm" and "layernorm". Defaults to "layernorm".
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        utt_emb_dim: int = 0,
        norm_type: NORM_TYPES = "rmsnorm",
    ):
        super().__init__()

        self.norm_type = norm_type

        self.block1 = Block1D(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            dropout=dropout,
            utt_emb_dim=utt_emb_dim,
            norm_type=norm_type,
        )
        self.block2 = Block1D(
            dim_out,
            dim_out,
            kernel_size=kernel_size,
            dropout=dropout,
            utt_emb_dim=utt_emb_dim,
            norm_type=norm_type,
        )

        self.res_conv = (
            torch.nn.Conv1d(dim_in, dim_out, 1)
            if dim_in != dim_out
            else torch.nn.Identity()
        )

    def forward(self, x, mask, utt_emb=None):
        """
        Forward pass of the ResnetBlock1D.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length).
            mask (torch.Tensor): Mask tensor of shape (batch_size, sequence_length).
            utt_emb (torch.Tensor, optional): Utterance embedding tensor of shape (batch_size, utt_emb_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, sequence_length).
        """
        h = self.block1(x, mask, utt_emb)
        h = self.block2(h, mask, utt_emb)
        output = h + self.res_conv(x * mask)
        return output


class AdaIN1d(nn.Module):
    def __init__(self, num_features, style_dim):
        super().__init__()
        """
        Adaptive Instance Normalization (AdaIN) layer.

        Args:
            num_features (int): Number of features in the condition sequence.
            style_dim (int): Dimension of the speaker embedding.
        """
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.context_mlp = nn.Linear(style_dim, 2 * num_features)

    def forward(self, x, s):
        if s.dim() == 3:
            s = s.squeeze(-1)

        h = self.context_mlp(s).unsqueeze(-1)
        gamma, beta = h.chunk(2, dim=1)

        return (1 + gamma) * self.norm(x) + beta


class ResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size: int = 3,
        actv: nn.Module = nn.SiLU(),
        normalize: bool = False,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.register_buffer("sqrt", torch.sqrt(torch.FloatTensor([0.5])).squeeze(0))
        self.dropout = nn.Dropout(p=dropout_p)

        self.conv1 = weight_norm(
            nn.Conv1d(
                dim_in,
                dim_in,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )
        self.conv2 = weight_norm(
            nn.Conv1d(
                dim_in,
                dim_out,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )

        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)

        self.shortcut = (
            weight_norm(nn.Conv1d(dim_in, dim_out, 1, padding=0, bias=False))
            if dim_in != dim_out
            else nn.Identity()
        )

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.dropout(x)
        x = self.conv1(x)

        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._residual(x) + self.shortcut(x)
        return x * self.sqrt


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

        self.W_scale = nn.Linear(utt_emb_dim, embedding_dim)
        self.W_bias = nn.Linear(utt_emb_dim, embedding_dim)

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


class FiLM(nn.Module):
    def __init__(self, in_dim, cond_dim):
        super().__init__()
        """
        FiLM layer.

        Args:
            in_dim (int): Input dimension of the phoneme sequence.
            cond_dim (int): Dimension of the condition sequence.
        """

        self.gain = nn.Conv1d(cond_dim, in_dim, kernel_size=1)
        self.bias = nn.Conv1d(cond_dim, in_dim, kernel_size=1)

        nn.init.xavier_uniform_(self.gain.weight)
        nn.init.constant_(self.gain.bias, 1)

        nn.init.xavier_uniform_(self.bias.weight)
        nn.init.constant_(self.bias.bias, 0)

    def forward(self, x, condition):
        gain = self.gain(condition)
        bias = self.bias(condition)

        return x * (1.0 + gain) + bias


class RMSNorm(nn.Module):
    def __init__(self, dim, unit_offset=False):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim**0.5

        self.g = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.g, 1.0 - float(unit_offset))

    def forward(self, x, *args):
        x = x.transpose(1, -1)
        gamma = self.g + float(self.unit_offset)
        x = F.normalize(x, dim=-1) * self.scale * gamma
        return x.transpose(1, -1)


class AdaptiveRMSNorm(nn.Module):
    def __init__(self, dim, dim_condition=None):
        super().__init__()
        self.scale = dim**0.5
        dim_condition = default(dim_condition, dim)

        self.to_gamma = nn.Linear(dim_condition, dim, bias=False)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, condition):
        x = x.transpose(1, -1)
        if condition.ndim == 2:
            condition = rearrange(condition, "b d -> b 1 d")

        normed = F.normalize(x, dim=-1)
        gamma = self.to_gamma(condition)
        x = normed * self.scale * (gamma + 1.0)
        return x.transpose(1, -1)


class LayerNorm(nn.Module):
    def __init__(self, channels, utt_emb=0, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x, *args):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class AdainResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hidden=None,
        kernel_size=3,
        style_dim=64,
        actv=nn.SiLU(),
        dropout_p=0.0,
    ):
        super().__init__()

        if dim_hidden is None:
            dim_hidden = dim_out
        self.actv = actv
        self.learned_sc = dim_in != dim_out
        self.dropout = nn.Dropout(dropout_p)
        self.register_buffer("sqrt", torch.sqrt(torch.FloatTensor([0.5])).squeeze(0))

        self.conv1 = weight_norm(
            nn.Conv1d(
                dim_in,
                dim_hidden,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )
        self.conv2 = weight_norm(
            nn.Conv1d(
                dim_hidden,
                dim_out,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )
        self.norm1 = AdaIN1d(dim_in, style_dim)
        self.norm2 = AdaIN1d(dim_hidden, style_dim)
        self.shortcut = (
            nn.Conv1d(dim_in, dim_out, 1, padding=0, bias=False)
            if dim_in != dim_out
            else nn.Identity()
        )

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x

    def forward(self, x, s):
        x = self.shortcut(x) + self._residual(x, s)
        return x * self.sqrt


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        utt_emb_dim: int = 0,
        p_dropout: float = 0.0,
        use_grn: bool = False,
        norm_type: NORM_TYPES = "rmsnorm",
    ):
        super().__init__()

        self.dwconv = nn.Conv1d(
            in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels
        )

        self.norm = get_normalization(norm_type, in_channels, utt_emb_dim)
        self.pwconv = nn.Sequential(
            nn.Linear(in_channels, filter_channels),
            nn.GELU(),
            GRN(filter_channels) if use_grn else nn.Identity(),
            nn.Linear(filter_channels, in_channels),
            nn.Dropout(p_dropout),
        )

    def forward(self, x, x_mask, g=None) -> torch.Tensor:
        h = self.dwconv(x) * x_mask
        h = self.norm(h, g).transpose(1, 2)
        h = self.pwconv(h).transpose(1, 2)
        x = h + x
        return x * x_mask


class ConvNeXt(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        n_layers: int = 1,
        utt_emb_dim: int = 0,
        p_dropout: float = 0.0,
        use_grn: bool = False,
        out_channels: int = None,
        norm_type: NORM_TYPES = "rmsnorm",
    ):
        super().__init__()

        # convnext blocks
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(
                ConvNeXtBlock(
                    in_channels=in_channels,
                    filter_channels=filter_channels,
                    utt_emb_dim=utt_emb_dim,
                    p_dropout=p_dropout,
                    use_grn=use_grn,
                    norm_type=norm_type,
                )
            )

        # final norm
        self.final_norm = get_normalization("rmsnorm", in_channels)

        # final conv
        self.final_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if out_channels
            else nn.Identity()
        )

    def forward(self, x, x_mask, utt_emb=None) -> torch.Tensor:
        # convnext blocks
        for layer in self.layers:
            x = layer(x, x_mask, utt_emb)

        # final layer norm
        x = self.final_norm(x * x_mask)

        # final conv
        x = self.final_conv(x)

        return x


class DepthwiseConv(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()

        self.dw_conv1d = nn.Conv1d(
            dim, dim, kernel_size, groups=dim, padding=kernel_size // 2
        )
        self.activation = nn.SiLU()

    def forward(self, x, x_mask):
        x = self.dw_conv1d(x * x_mask)
        x = self.activation(x)
        return x * x_mask


class Conv1dGLU(nn.Module):
    """
    Conv1d + GLU(Gated Linear Unit) with residual connection.
    For GLU refer to https://arxiv.org/abs/1612.08083 paper.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(Conv1dGLU, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(
            in_channels,
            2 * out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = residual + self.dropout(x)

        return x


class BSConv1d(nn.Module):
    """https://arxiv.org/pdf/2003.13549.pdf"""

    def __init__(self, channels_in: int, channels_out: int, kernel_size: int):
        super().__init__()
        self.pointwise = nn.Conv1d(channels_in, channels_out, kernel_size=1)
        self.depthwise = nn.Conv1d(
            channels_out,
            channels_out,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels_out,
        )

    def forward(self, x, x_mask):
        x1 = self.pointwise(x * x_mask)
        x2 = self.depthwise(x1)
        return x2 * x_mask


class ConditionalConv1dGLU(nn.Module):
    """From DeepVoice 3"""

    def __init__(self, hidden_channels: int, kernel_size: int, utt_emb_dim: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv1d(
            hidden_channels,
            2 * hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.embedding_proj = nn.Conv1d(utt_emb_dim, hidden_channels, 1)
        self.softsign = torch.nn.Softsign()

    def forward(self, x, x_mask, utt_emb):
        h = self.conv(x) * x_mask
        a, b = h.split(self.hidden_channels, dim=1)

        # softsign utt emb
        embeddings = self.embedding_proj(utt_emb)
        softsign = self.softsign(embeddings)
        a = a + softsign.unsqueeze(-1)
        # GLU
        x = a * torch.sigmoid(b)
        return x * x_mask


class Downsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        elif self.layer_type == "timepreserve":
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == "half":
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError(
                "Got unexpected donwsampletype %s, expected is [none, timepreserve, half]"
                % self.layer_type
            )


class Wav2Vec2StackedPositionEncoder(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        kernel_size: int,
        groups: int,
        p_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.Conv1d(
                            dim,
                            dim,
                            kernel_size,
                            padding="same",
                            groups=groups,
                        ),
                        get_normalization("rmsnorm", dim),
                        nn.GELU(),
                        nn.Dropout(p_dropout),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for conv, norm, activation, drop in self.layers:
            x = conv(x * mask)
            x = norm(x)
            x = activation(x)
            x = drop(x)

        return x * mask


class AdaLNZero(nn.Module):
    def __init__(self, dim, dim_condition, init_bias_value=-2.0):
        super().__init__()
        self.to_gamma = nn.Linear(dim_condition, dim)

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.constant_(self.to_gamma.bias, init_bias_value)

    def forward(self, x, condition):
        condition = self.to_gamma(condition).sigmoid()
        condition = rearrange(condition, "b d -> b d 1")

        return x * condition
