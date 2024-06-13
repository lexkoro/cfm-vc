import math

import torch
from einops import rearrange
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F

from modules import commons
from modules.modules import AdaIN1d, ConditionalLayerNorm, FiLM, LayerNorm


class ConditioningEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        n_heads,
        dim_head=None,
        kernel_size=1,
        p_dropout=0.1,
        cond_emb_dim=None,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_heads = n_heads
        self.kernel_size = kernel_size

        if hidden_channels != cond_emb_dim:
            self.cond_conv = nn.Conv1d(
                cond_emb_dim, hidden_channels, kernel_size=kernel_size
            )
        else:
            self.cond_conv = nn.Identity()

        self.attn = MultiHeadAttention(
            hidden_channels,
            hidden_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            p_dropout=p_dropout,
        )
        self.film = FiLM(hidden_channels, hidden_channels)

    def forward(self, x, x_mask, cond_latent=None, cond_mask=None):
        cond_latent = self.cond_conv(cond_latent)

        # attn mask
        attn_mask = cond_mask.unsqueeze(2) * x_mask.unsqueeze(-1)

        # cross attn
        y = self.attn(x, cond_latent, attn_mask)

        # film
        x = self.film(x, y)

        return x * x_mask


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        dim_head=None,
        kernel_size=1,
        p_dropout=0.0,
        causal_ffn=False,
        utt_emb_dim=512,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.encdec_attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.self_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    dim_head=dim_head,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_0.append(
                ConditionalLayerNorm(hidden_channels, utt_emb_dim)
            )
            self.encdec_attn_layers.append(
                ConditioningEncoder(
                    hidden_channels,
                    n_heads,
                    dim_head=dim_head,
                    p_dropout=p_dropout,
                    cond_emb_dim=hidden_channels,
                )
            )
            self.norm_layers_1.append(
                ConditionalLayerNorm(hidden_channels, utt_emb_dim)
            )
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                    causal=causal_ffn,
                )
            )
            self.norm_layers_2.append(
                ConditionalLayerNorm(hidden_channels, utt_emb_dim)
            )

    def forward(self, x, x_mask, h, h_mask, g=None):
        """
        x: decoder input
        h: encoder output
        """
        self_attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y, g)

            y = self.encdec_attn_layers[i](x, x_mask, h, h_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y, g)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y, g)
        x = x * x_mask
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        dim_head=None,
        kernel_size=1,
        p_dropout=0.0,
        utt_emb_dim=0,
        use_cond_norm=True,
        causal_ffn=False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()

        self.norm_layers_1 = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        self.ffn_layers_1 = nn.ModuleList()

        for i in range(self.n_layers):
            self.attn_layers.append(
                ConditionalMultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    dim_head=dim_head,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_1.append(
                ConditionalLayerNorm(hidden_channels, utt_emb_dim)
                if use_cond_norm and utt_emb_dim > 0
                else LayerNorm(hidden_channels)
            )

            self.ffn_layers_1.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size=kernel_size,
                    p_dropout=p_dropout,
                    causal=causal_ffn,
                )
            )
            self.norm_layers_2.append(
                ConditionalLayerNorm(hidden_channels, utt_emb_dim)
                if use_cond_norm and utt_emb_dim > 0
                else LayerNorm(hidden_channels)
            )

        self.final_norm = LayerNorm(hidden_channels)

    def forward(self, x, x_mask, h, h_mask, g=None):
        # attn mask
        cond_mask = torch.cat([h_mask, x_mask], dim=-1)
        attn_mask = cond_mask.unsqueeze(2) * cond_mask.unsqueeze(-1)

        x = x * x_mask

        for i in range(self.n_layers):
            # self-attention
            self_in = self.norm_layers_1[i](x, g)
            x = self.attn_layers[i](self_in, c=h, attn_mask=attn_mask) + x

            # feed-forward
            ff_in = self.norm_layers_2[i](x, g)
            x = self.ffn_layers_1[i](ff_in, x_mask) + x

        x = self.final_norm(x) * x_mask
        return x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @property
    def device(self):
        return next(self.buffers()).device

    @autocast(enabled=False)
    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@autocast(enabled=False)
def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        dim_head=None,
        p_dropout=0.0,
        proximal_bias=False,
        proximal_init=False,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        if dim_head is not None:
            self.dim_head = dim_head
            self.channels = dim_head * n_heads
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.proximal_bias = proximal_bias
        self.p_dropout = p_dropout
        self.attn = None

        self.k_channels = self.channels // n_heads

        self.conv_q = nn.Conv1d(channels, self.channels, 1)
        self.conv_k = nn.Conv1d(channels, self.channels, 1)
        self.conv_v = nn.Conv1d(channels, self.channels, 1)

        self.query_rotary_pe = RotaryEmbedding(self.k_channels)
        self.key_rotary_pe = RotaryEmbedding(self.k_channels)

        self.conv_o = nn.Conv1d(self.channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(
            q,
            k,
            v,
            mask=attn_mask,
        )

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = rearrange(query, "b (h c) t-> b h t c", h=self.n_heads)
        key = rearrange(key, "b (h c) t-> b h t c", h=self.n_heads)
        value = rearrange(value, "b (h c) t-> b h t c", h=self.n_heads)

        query_emb = self.query_rotary_pe(t_t)
        query = apply_rotary_pos_emb(query_emb, query)

        key_emb = self.key_rotary_pe(t_s)
        key = apply_rotary_pos_emb(key_emb, key)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)

        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device, dtype=scores.dtype
            )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    @staticmethod
    def _attention_bias_proximal(length):
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class ConditionalMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        dim_head=None,
        p_dropout=0.0,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        if dim_head is not None:
            self.dim_head = dim_head
            self.channels = dim_head * n_heads
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout

        self.k_channels = self.channels // n_heads

        self.conv_q = torch.nn.Conv1d(channels, self.channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, self.channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, self.channels, 1)

        self.query_rotary_pe = RotaryEmbedding(self.k_channels)
        self.key_rotary_pe = RotaryEmbedding(self.k_channels)

        self.conv_o = torch.nn.Conv1d(self.channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        # length of the sequence
        t_t, t_s = x.size(2), c.size(2)

        # rotary positional embedding
        query_emb = self.query_rotary_pe(t_t)
        key_emb = self.key_rotary_pe(t_s)
        query_emb = key_emb = torch.cat((key_emb, query_emb), dim=0)
        x = c = torch.cat((c, x), dim=-1)
        t_t = x.size(2)

        # conv
        query = self.conv_q(x)
        key = self.conv_k(c)
        value = self.conv_v(c)

        # split heads
        b, d = key.size(0), key.size(1)
        query = rearrange(query, "b (h c) t-> b h t c", h=self.n_heads)
        key = rearrange(key, "b (h c) t-> b h t c", h=self.n_heads)
        value = rearrange(value, "b (h c) t-> b h t c", h=self.n_heads)

        # apply rotary positional embedding
        query = apply_rotary_pos_emb(query_emb, query)
        key = apply_rotary_pos_emb(key_emb, key)

        # attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)

        x = self.conv_o(output)
        x = x[:, :, t_s:]
        return x


class FFN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout=0.0,
        causal=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = nn.GELU(approximate="tanh")
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        x = self.activation(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x


class ConditionalFFN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        style_dim,
        kernel_size,
        p_dropout=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = nn.GELU("tanh")

        self.conv_1 = nn.Conv1d(
            in_channels,
            filter_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.conv_2 = nn.Conv1d(
            filter_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )

        self.norm1 = AdaIN1d(style_dim, in_channels)
        self.norm2 = AdaIN1d(style_dim, filter_channels)

        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask, s):
        x = self.norm1(x, s)
        x = self.conv_1(x * x_mask)
        x = self.activation(x)
        x = self.drop(x)
        x = self.norm2(x, s)
        x = self.conv_2(x * x_mask)
        return x * x_mask
