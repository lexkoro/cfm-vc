from typing import List, Optional

import torch
import torch.nn as nn

from modules import Wav2Vec2StackedPositionEncoder
from modules.attention.transformer import Encoder


class AttentionPooling(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(AttentionPooling, self).__init__()

        self.attention = nn.Sequential(
            nn.Conv1d(hidden_channels, 128, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(128, hidden_channels, kernel_size=1),
        )

        # fc
        self.fc = nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, x, mask):
        # apply attention
        attn_weights = self.attention(x)
        attn_weights = attn_weights.masked_fill(mask == 0, -1e4)
        attn_weights = torch.softmax(attn_weights, dim=2)

        # perform attention pooling
        pooled = self.asp_encoder(x, attn_weights, mask)

        # apply final linear layer
        pooled = self.fc(pooled)

        return pooled

    @staticmethod
    def asp_encoder(x, w, mask):
        len_ = mask.sum(dim=2)
        mu = torch.sum((x * w) * mask, dim=2) / len_
        sg = torch.sum(((x**2) * w) * mask, dim=2) / len_
        sg = torch.sqrt((sg - (mu**2)).clamp(min=1e-5))
        x = torch.cat((mu, sg), 1)
        return x


class SpeakerEncoder(nn.Module):
    """SpeakerEncoder for extracting speaker embedding from mel-spectrogram
    - Prenet: Conv1d + Wav2Vec2StackedPositionEncoder
    - Self-Attention: Transformer Encoder
    - ASP Pooling: Attention-based Statistical Pooling
    """

    def __init__(
        self,
        spec_channels,
        hidden_channels,
        filter_channels,
        n_heads: int = 2,
        dim_head: Optional[int] = None,
        n_layers: int = 2,
        p_dropout: float = 0.1,
        speaker_embedding: int = 256,
    ):
        super(SpeakerEncoder, self).__init__()

        # Prenet
        self.prenet = nn.Conv1d(spec_channels, hidden_channels, kernel_size=1)
        self.conv = Wav2Vec2StackedPositionEncoder(
            depth=2,
            dim=hidden_channels,
            kernel_size=15,
            groups=16,
            p_dropout=p_dropout,
        )

        # self attention
        self.encoder = Encoder(
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            n_layers=n_layers,
            p_dropout=p_dropout,
        )

        # asp pooling
        self.asp_encoder = AttentionPooling(hidden_channels, speaker_embedding)

    def forward(self, x, x_mask):
        # mel prenet
        x = self.prenet(x)
        x = self.conv(x, x_mask) + x

        # encoder
        x = self.encoder(x, x_mask)

        # asp pooling
        spk = self.asp_encoder(x, x_mask)

        return spk
