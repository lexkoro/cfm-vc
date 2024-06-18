import math
from typing import Optional

from torch import nn

import modules.commons as commons
from modules.attentions import Decoder


class PPGDecoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_feats: int,
        in_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        dim_head: Optional[int] = None,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        utt_emb_dim: int = 0,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        # decoder
        self.ppg_decoder = Decoder(
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            causal_ffn=True,
            utt_emb_dim=utt_emb_dim,
        )

        # project to mu
        self.proj_m = nn.Conv1d(hidden_channels, n_feats, kernel_size=1)

    def forward(
        self,
        x,
        x_lengths,
        cond,
        cond_mask,
        speaker_embedding,
    ):
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x = x.transpose(1, -1)  # [b, h, t]
        x_mask = commons.sequence_mask(x_lengths, x.size(2)).unsqueeze(1).to(x.dtype)

        # encoder
        x = self.ppg_decoder(x, x_mask, cond, cond_mask, speaker_embedding)

        # project to mu
        mu_x = self.proj_m(x) * x_mask

        return x, mu_x, x_mask
