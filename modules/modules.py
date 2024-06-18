import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from modules.wavenet.wavenet import WaveNet

LRELU_SLOPE = 0.1


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

    def __init__(self, dim, dim_out, dropout=0.1, utt_emb_dim=0, norm_type="layernorm"):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm_type = norm_type

        if norm_type == "condlayernorm" and utt_emb_dim != 0:
            self.norm = ConditionalLayerNorm(dim_out, utt_emb_dim)
        elif norm_type == "layernorm":
            self.norm = LayerNorm(dim_out)
        else:
            raise NotImplementedError("norm_type not implemented.")

        self.mish = nn.Mish()
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
        output = self.norm(output, utt_emb)
        output = self.mish(output)
        output = self.dropout(output)
        return output * mask


class ResnetBlock1D(torch.nn.Module):
    """
    Residual block for 1D ResNet.

    Args:
        dim_in (int): Number of input channels.
        dim_out (int): Number of output channels.
        dropout (float, optional): Dropout rate. Default is 0.1.
        utt_emb_dim (int, optional): Dimension of utterance embedding. Default is 512.
        norm_type (str, optional): The type of normalization to be applied.
            Possible values are "condlayernorm", "condgroupnorm" and "layernorm". Defaults to "layernorm".
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dropout: float = 0.1,
        utt_emb_dim: int = 512,
        norm_type: str = "layernorm",
    ):
        super().__init__()

        self.norm_type = norm_type

        self.block1 = Block1D(
            dim_in,
            dim_out,
            dropout=dropout,
            utt_emb_dim=utt_emb_dim,
            norm_type=norm_type,
        )
        self.block2 = Block1D(
            dim_out,
            dim_out,
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


class FiLM(nn.Module):
    def __init__(self, in_dim, cond_dim):
        super().__init__()

        self.gain = nn.Conv1d(cond_dim, in_dim, kernel_size=1)
        self.bias = nn.Conv1d(cond_dim, in_dim, kernel_size=1)

        nn.init.xavier_uniform_(self.gain.weight)
        nn.init.constant_(self.gain.bias, 1)

        nn.init.xavier_uniform_(self.bias.weight)
        nn.init.constant_(self.bias.bias, 0)

    def forward(self, x, condition):
        gain = self.gain(condition)
        bias = self.bias(condition)

        return x * gain + bias


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super(Upsample1d, self).__init__()
        self.conv = weight_norm(
            torch.nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.conv(x)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super(Downsample1d, self).__init__()
        self.conv = weight_norm(
            torch.nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.conv(x)


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


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x, *args):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class ConvReluNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size,
        n_layers,
        p_dropout,
        eps=1e-5,
    ):
        super(ConvReluNorm, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.eps = eps

        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(
            torch.nn.Conv1d(
                in_channels, hidden_channels, kernel_size, padding=kernel_size // 2
            )
        )
        self.relu_drop = torch.nn.Sequential(
            torch.nn.ReLU(), torch.nn.Dropout(p_dropout)
        )
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                torch.nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
        self.proj = torch.nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.instance_norm(x, x_mask)
            x = self.relu_drop(x)
        x = self.proj(x)
        return x * x_mask

    def instance_norm(self, x, mask, return_mean_std=False):
        mean, std = self.calc_mean_std(x, mask)
        x = (x - mean) / std
        if return_mean_std:
            return x, mean, std
        else:
            return x

    def calc_mean_std(self, x, mask=None):
        x = x * mask
        B, C = x.shape[:2]
        mn = x.view(B, C, -1).mean(-1)
        sd = (x.view(B, C, -1).var(-1) + self.eps).sqrt()
        mn = mn.view(B, C, *((len(x.shape) - 2) * [1]))
        sd = sd.view(B, C, *((len(x.shape) - 2) * [1]))
        return mn, sd


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv1d(
            dim, hidden_dim * 3, 1, bias=False
        )  # NOTE: 1x1 conv
        self.to_out = torch.nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, d = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) d -> qkv b heads c d", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(out, "b heads c d -> b (heads c) d", heads=self.heads, d=d)
        return self.to_out(out)


class ResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in,
        intermediate_dim,
        dim_out,
        kernel_size=3,
        actv=nn.LeakyReLU(0.2),
        normalize=False,
        dropout_p=0.2,
    ):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.register_buffer("sqrt", torch.sqrt(torch.FloatTensor([0.5])).squeeze(0))
        self.dropout = nn.Dropout(p=dropout_p)

        self.conv1 = weight_norm(
            nn.Conv1d(
                dim_in,
                intermediate_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )
        self.conv2 = weight_norm(
            nn.Conv1d(
                intermediate_dim,
                dim_out,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )

        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(intermediate_dim, affine=True)

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


class Log(nn.Module):
    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class ElementwiseAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hidden=None,
        kernel_size=3,
        style_dim=64,
        actv=nn.LeakyReLU(0.2),
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
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_hidden)
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
