import torch
from torch import nn

import modules.attentions as attentions
from modules.modules import ResBlk1d
from modules.variance_decoder import AuxDecoder, ConditionalEmbedding
from utils import f0_to_coarse, normalize_f0


class ContentEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        n_feats,
        ssl_dim,
        ppgs_dim,
        kernel_size,
        n_layers,
        filter_channels=None,
        n_heads=None,
        p_dropout=0.0,
        utt_emb_dim=0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.uv_emb = nn.Embedding(2, hidden_channels)
        self.f0_emb = ConditionalEmbedding(256, hidden_channels, style_dim=utt_emb_dim)

        # ContentVec prenet
        self.ssl_prenet = ResBlk1d(
            dim_in=ssl_dim,
            intermediate_dim=384,
            dim_out=hidden_channels,
            kernel_size=5,
            normalize=False,
        )

        # f0 decoder
        self.f0_decoder = AuxDecoder(
            input_channels=1,
            hidden_channels=hidden_channels,
            output_channels=1,
            kernel_size=kernel_size,
            n_layers=n_layers,
            n_heads=n_heads,
            p_dropout=0.1,
            utt_emb_dim=utt_emb_dim,
        )

        # ppg decoder
        self.ppg_decoder = AuxDecoder(
            input_channels=ppgs_dim,
            hidden_channels=hidden_channels,
            output_channels=hidden_channels,
            kernel_size=kernel_size,
            n_layers=n_layers,
            n_heads=n_heads,
            p_dropout=0.1,
            utt_emb_dim=utt_emb_dim,
        )

        # encoder
        self.encoder = attentions.Encoder(
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            utt_emb_dim=utt_emb_dim,
        )

        # project to mu
        self.proj_m = ResBlk1d(
            dim_in=hidden_channels,
            intermediate_dim=hidden_channels,
            dim_out=n_feats,
            kernel_size=3,
            normalize=True,
        )

    def forward(
        self,
        x,
        x_mask,
        cond=None,
        cond_mask=None,
        f0=None,
        uv=None,
        ppgs=None,
        utt_emb=None,
    ):
        # prenet
        x = self.ssl_prenet(x) * x_mask

        # add uv to x
        x = x + self.uv_emb(uv.long()).transpose(1, 2)

        # ppg decoder
        ppg_pred = self.ppg_decoder(x, x_mask, ppgs, cond, cond_mask, utt_emb)

        # pitch
        lf0 = 2595.0 * torch.log10(1.0 + f0 / 700.0) / 500
        f0_norm = normalize_f0(lf0, x_mask, uv)
        f0_pred = self.f0_decoder(x, x_mask, f0_norm, cond, cond_mask, utt_emb)
        f0 = f0_to_coarse(f0.squeeze(1))
        f0_emb = self.f0_emb(f0, x_mask, utt_emb)

        # add f0 and ppg to x
        x = x + f0_emb + ppg_pred

        # encode prosodic features
        x = self.encoder(x, x_mask, utt_emb)

        # # project to mu
        mu = self.proj_m(x) * x_mask

        return mu, x_mask, f0_pred, lf0

    def vc(
        self,
        x,
        x_mask,
        cond,
        cond_mask,
        f0=None,
        uv=None,
        ppgs=None,
        utt_emb=None,
    ):
        # prenet
        x = self.ssl_prenet(x) * x_mask

        # add uv to x
        x = x + self.uv_emb(uv.long()).transpose(1, 2)

        # ppg decoder
        ppg_pred = self.ppg_decoder(x, x_mask, ppgs, cond, cond_mask, utt_emb)

        # pitch
        lf0 = 2595.0 * torch.log10(1.0 + f0 / 700.0) / 500
        f0_norm = normalize_f0(lf0, x_mask, uv)
        f0_pred = self.f0_decoder(x, x_mask, f0_norm, cond, cond_mask, utt_emb)
        f0 = (700 * (torch.pow(10, f0_pred * 500 / 2595) - 1)).squeeze(1)
        f0 = f0_to_coarse(f0)
        f0_emb = self.f0_emb(f0, x_mask, utt_emb)

        # add f0 and ppg to x
        x = x + f0_emb + ppg_pred

        # encode prosodic features
        x = self.encoder(x, x_mask, utt_emb)

        # # project to mu
        mu = self.proj_m(x) * x_mask

        return mu, x_mask
