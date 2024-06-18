import torch
import torch.nn as nn

import modules.commons as commons
from modules.attentions import MultiHeadAttention
from modules.perceiver_encoder import PerceiverResampler


class AttentionPooling(nn.Module):
    def __init__(self, hidden_channels):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(hidden_channels, 128, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(128, hidden_channels, kernel_size=1),
        )

    def forward(self, x, mask):
        attn_weights = self.attention(x)
        attn_weights = attn_weights.masked_fill(mask == 0, -1e4)
        attn_weights = torch.softmax(attn_weights, dim=2)
        pooled = self.asp_encoder(x, attn_weights, mask)
        return pooled

    @staticmethod
    def asp_encoder(x, w, mask):
        len_ = mask.sum(dim=2)
        mu = torch.sum((x * w) * mask, dim=2) / len_
        sg = torch.sum(((x**2) * w) * mask, dim=2) / len_
        sg = torch.sqrt((sg - (mu**2)).clamp(min=1e-5))
        x = torch.cat((mu, sg), 1)
        return x


class Conv1dGLU(nn.Module):
    """
    Conv1d + GLU(Gated Linear Unit) with residual connection.
    For GLU refer to https://arxiv.org/abs/1612.08083 paper.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(Conv1dGLU, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(
            in_channels, 2 * out_channels, kernel_size=kernel_size, padding=2
        )
        self.p_dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = residual + self.p_dropout(x)
        return x


class MelStyleEncoder(nn.Module):
    """MelStyleEncoder"""

    def __init__(
        self,
        in_channels=80,
        hidden_channels=192,
        utt_channels=512,
        kernel_size=5,
        p_dropout=0.0,
        n_heads=2,
        dim_head=64,
    ):
        super(MelStyleEncoder, self).__init__()

        # encode
        self.spectral = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            nn.Mish(),
            nn.Dropout(p_dropout),
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.Mish(),
            nn.Dropout(p_dropout),
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(hidden_channels, hidden_channels, kernel_size, p_dropout),
            Conv1dGLU(hidden_channels, hidden_channels, kernel_size, p_dropout),
        )

        # perceiver resampler
        self.resampler = PerceiverResampler(
            hidden_channels=hidden_channels,
            depth=2,
            num_latents=32,
            dim_head=dim_head,
            heads=n_heads,
            ff_mult=4,
            p_dropout=p_dropout,
        )

        # self attn
        self.slf_attn = MultiHeadAttention(
            hidden_channels,
            hidden_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            p_dropout=p_dropout,
        )
        self.attn_drop = nn.Dropout(p_dropout)

        # attention pooling
        self.attention_pooling = AttentionPooling(hidden_channels)

        # fc
        self.fc = nn.Linear(hidden_channels * 2, utt_channels)

    def forward(self, x, x_mask=None):
        # spectral
        x = self.spectral(x) * x_mask

        # temporal
        x = self.temporal(x) * x_mask

        # resampler
        latents, latents_mask = self.resampler(x, x_mask)
        # attention
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        y = self.slf_attn(x, c=x, attn_mask=attn_mask)
        x = self.attn_drop(y) + x

        # attention pooling
        x = self.attention_pooling(x, x_mask)

        # fc
        utt_emb = self.fc(x)

        return utt_emb, latents, latents_mask
