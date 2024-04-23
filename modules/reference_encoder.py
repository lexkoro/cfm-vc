import torch
import torch.nn as nn

from modules.attentions import Encoder
from modules.mel_encoder import MelEncoder


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

        # attn
        self.attn = Encoder(
            hidden_channels=hidden_channels,
            filter_channels=hidden_channels * 4,
            n_layers=2,
            n_heads=n_heads,
            dim_head=dim_head,
            kernel_size=3,
            p_dropout=0.1,
        )

        self.fc = nn.Conv1d(hidden_channels, utt_channels, kernel_size=1)

    def temporal_avg_pool(self, x, mask=None):
        # avg pooling
        len_ = mask.sum(dim=2)
        x = x.sum(dim=2)
        out = torch.div(x, len_)
        return out

    def forward(self, x, x_mask=None):
        # spectral
        x = self.spectral(x) * x_mask
        # temporal
        x = self.temporal(x) * x_mask

        # attention
        x = self.attn(x, x_mask)

        # fc
        x = self.fc(x) * x_mask

        # temoral average pooling for utterance embedding
        utt_emb = self.temporal_avg_pool(x, mask=x_mask)

        return utt_emb
