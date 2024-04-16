import torch
import torch.nn as nn

from modules.mel_encoder import MelEncoder
from modules.perceiver_encoder import PerceiverResampler


class MelStyleEncoder(nn.Module):
    """MelStyleEncoder"""

    def __init__(
        self,
        in_channels=80,
        hidden_channels=256,
        cond_channels=192,
        utt_channels=512,
        kernel_size=5,
        n_heads=2,
        dim_head=None,
    ):
        super(MelStyleEncoder, self).__init__()

        # encoder
        self.encoder = MelEncoder(
            in_channels=in_channels,
            out_channels=hidden_channels,
            hidden_channels=cond_channels,
            kernel_size=kernel_size,
            dilation_rate=1,
            n_layers=16,
        )

        # perceiver encoder
        self.perceiver_encoder = PerceiverResampler(
            hidden_channels=hidden_channels,
            depth=2,
            num_latents=32,
            dim_head=dim_head,
            heads=n_heads,
            ff_mult=4,
        )

        self.cond_proj = nn.Conv1d(hidden_channels, cond_channels, kernel_size=1)
        self.utt_proj = nn.Conv1d(hidden_channels, utt_channels, kernel_size=1)

    def temporal_avg_pool(self, x, mask=None):
        # avg pooling
        len_ = mask.sum(dim=-1)
        x = torch.sum(x * mask, dim=-1) / len_
        return x

    def forward(self, x, x_mask=None):
        # encode mel (x)
        encoded_mel = self.encoder(x, x_mask)

        # perceiver encoder
        cond, cond_mask = self.perceiver_encoder(encoded_mel, x_mask)

        # project to cond and utt embeddings
        utt_emb = self.utt_proj(cond) * cond_mask
        cond = self.cond_proj(cond) * cond_mask

        # temoral average pooling for utterance embedding
        utt_emb = self.temporal_avg_pool(utt_emb, mask=cond_mask)

        return utt_emb, cond, cond_mask
