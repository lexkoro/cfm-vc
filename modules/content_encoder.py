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
        ssl_dim,
        kernel_size,
        n_layers,
        filter_channels=None,
        n_heads=None,
        dim_head=None,
        p_dropout=0.0,
        utt_emb_dim=0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.uv_emb = nn.Embedding(2, hidden_channels)
        self.f0_emb = ConditionalEmbedding(256, hidden_channels, style_dim=utt_emb_dim)
        self.energy_emb = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=9, padding=(9 - 1) // 2),
            nn.Dropout(0.5),
        )

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
            dim_head=dim_head,
            p_dropout=p_dropout,
            utt_emb_dim=utt_emb_dim,
        )

        # energy decoder
        self.energy_decoder = AuxDecoder(
            input_channels=1,
            hidden_channels=hidden_channels,
            output_channels=1,
            kernel_size=kernel_size,
            n_layers=n_layers // 2,
            n_heads=n_heads,
            dim_head=dim_head,
            p_dropout=p_dropout,
            utt_emb_dim=utt_emb_dim,
        )

        # encoder
        self.encoder = attentions.Encoder(
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            utt_emb_dim=utt_emb_dim,
        )

    def forward(
        self,
        x,
        x_mask,
        cond=None,
        cond_mask=None,
        f0=None,
        uv=None,
        energy=None,
        utt_emb=None,
    ):
        # prenet
        x = self.ssl_prenet(x) * x_mask

        x_speaker_classifier = x

        # add uv to x
        x = x + self.uv_emb(uv.long()).transpose(1, 2)

        # pitch
        lf0 = 2595.0 * torch.log10(1.0 + f0 / 700.0) / 500
        f0_norm = normalize_f0(lf0, x_mask, uv)
        f0_pred = self.f0_decoder(x, x_mask, f0_norm, cond, cond_mask, utt_emb)
        f0 = f0_to_coarse(f0.squeeze(1))
        f0_emb = self.f0_emb(f0, x_mask, utt_emb)

        # energy
        energy_pred = self.energy_decoder(x, x_mask, energy, cond, cond_mask, utt_emb)
        energy_emb = self.energy_emb(energy)

        # add f0 to x
        x = x + f0_emb + energy_emb

        # encode prosodic features
        x = self.encoder(x, x_mask, cond, cond_mask, utt_emb)

        return x_speaker_classifier, x, f0_pred, lf0, energy_pred

    def vc(
        self,
        x,
        x_mask,
        cond,
        cond_mask,
        f0=None,
        uv=None,
        energy=None,
        utt_emb=None,
    ):
        # prenet
        x = self.ssl_prenet(x) * x_mask

        # add uv to x
        x = x + self.uv_emb(uv.long()).transpose(1, 2)

        # pitch
        lf0 = 2595.0 * torch.log10(1.0 + f0 / 700.0) / 500
        f0_norm = normalize_f0(lf0, x_mask, uv)
        f0_pred = self.f0_decoder(x, x_mask, f0_norm, cond, cond_mask, utt_emb)
        f0 = (700 * (torch.pow(10, f0_pred * 500 / 2595) - 1)).squeeze(1)
        f0 = f0_to_coarse(f0)
        f0_emb = self.f0_emb(f0, x_mask, utt_emb)

        # energy
        energy_pred = self.energy_decoder(x, x_mask, energy, cond, cond_mask, utt_emb)
        energy_emb = self.energy_emb(energy_pred)

        # add f0 to x
        x = x + f0_emb + energy_emb

        # encode prosodic features
        x = self.encoder(x, x_mask, cond, cond_mask, utt_emb)

        return x
