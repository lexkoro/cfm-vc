import math

import torch
from torch import nn

import modules.attentions as attentions
import modules.commons as commons
from modules.cfm.cfm_neuralode import ConditionalFlowMatching
from modules.modules import ResBlk1d
from modules.reference_encoder import MelStyleEncoder
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
        p_dropout=None,
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
            kernel_size=3,
            n_layers=6,
            n_heads=2,
            p_dropout=0.1,
            utt_emb_dim=512,
        )

        # ppg decoder
        self.ppg_decoder = AuxDecoder(
            input_channels=ppgs_dim,
            hidden_channels=hidden_channels,
            output_channels=hidden_channels,
            kernel_size=3,
            n_layers=6,
            n_heads=2,
            p_dropout=0.1,
            utt_emb_dim=512,
        )

        # decoder
        self.encoder = attentions.Encoder(
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=2,
            n_layers=6,
            kernel_size=3,
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
        ppg = self.ppg_decoder(x, x_mask, ppgs, cond, cond_mask, utt_emb)

        # pitch
        lf0 = 2595.0 * torch.log10(1.0 + f0 / 700.0) / 500
        f0_norm = normalize_f0(lf0, x_mask, uv)
        f0_pred = self.f0_decoder(x, x_mask, f0_norm, cond, cond_mask, utt_emb)
        f0 = f0_to_coarse(f0.squeeze(1))
        f0_emb = self.f0_emb(f0, x_mask, utt_emb)

        # add f0 and ppg to x
        x = x + f0_emb + ppg

        # encode prosodic features
        x = self.encoder(x, x_mask, utt_emb)

        # # project to mu
        mu = self.proj_m(x) * x_mask

        return mu, x_mask, f0_pred, lf0

    def vc(
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
        ppg = self.ppg_decoder(x, x_mask, ppgs, cond, cond_mask, utt_emb)

        # pitch
        lf0 = 2595.0 * torch.log10(1.0 + f0 / 700.0) / 500
        f0_norm = normalize_f0(lf0, x_mask, uv)
        f0_pred = self.f0_decoder(x, x_mask, f0_norm, cond, cond_mask, utt_emb)
        f0 = (700 * (torch.pow(10, f0_pred * 500 / 2595) - 1)).squeeze(1)
        f0 = f0_to_coarse(f0)
        f0_emb = self.f0_emb(f0, x_mask, utt_emb)

        # add f0 and ppg to x
        x = x + f0_emb + ppg

        # encode prosodic features
        x = self.encoder(x, x_mask, utt_emb)

        # # project to mu
        mu = self.proj_m(x) * x_mask

        return mu, x_mask


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        speaker_embedding,
        n_speakers,
        ssl_dim,
        ppgs_dim,
        **kwargs,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.speaker_embedding = speaker_embedding
        self.n_speakers = n_speakers
        self.ssl_dim = ssl_dim
        self.ppgs_dim = ppgs_dim

        # content encoder
        self.enc_p = ContentEncoder(
            hidden_channels=hidden_channels,
            n_feats=spec_channels,
            ssl_dim=ssl_dim,
            ppgs_dim=ppgs_dim,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            utt_emb_dim=speaker_embedding,
        )

        # reference mel encoder
        self.mel_encoder = MelStyleEncoder(
            in_channels=spec_channels,
            hidden_channels=256,
            cond_channels=hidden_channels,
            utt_channels=speaker_embedding,
            kernel_size=5,
            n_heads=8,
            dim_head=64,
        )

        # conditional flow matching decoder
        self.decoder = ConditionalFlowMatching(
            in_channels=self.spec_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.spec_channels,
            spk_emb_dim=speaker_embedding,
            estimator="dit",
        )

    def forward(self, c, f0, uv, spec, ppgs=None, c_lengths=None):
        # if self.n_speakers > 1 and self.speaker_embedding:
        #     g = F.normalize(g).unsqueeze(-1)

        # x_mask
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(
            c.dtype
        )

        # reference mel encoder
        g, cond, cond_mask = self.mel_encoder(spec, x_mask)

        # content encoder
        mu_y, x_mask, f0_pred, lf0 = self.enc_p(
            c,
            x_mask,
            cond=cond,
            cond_mask=cond_mask,
            f0=f0,
            uv=uv,
            ppgs=ppgs,
            utt_emb=g,
        )

        # Compute loss of score-based decoder
        diff_loss, _ = self.decoder.forward(
            spec, None, x_mask, mu_y, spk=g, cond=cond, cond_mask=cond_mask
        )

        prior_loss = torch.sum(
            0.5 * ((spec - mu_y) ** 2 + math.log(2 * math.pi)) * x_mask
        )
        prior_loss = prior_loss / (torch.sum(x_mask) * self.spec_channels)

        return (prior_loss, diff_loss, f0_pred, lf0)

    @torch.no_grad()
    def infer(
        self,
        c,
        spec,
        f0,
        uv,
        ppgs,
        n_timesteps=10,
        temperature=1.0,
        guidance_scale=0.0,
    ):
        # if self.n_speakers > 1 and self.speaker_embedding:
        #     g = F.normalize(g).unsqueeze(-1)

        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)

        # x mask
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(
            c.dtype
        )

        # reference mel encoder
        g, cond, cond_mask = self.mel_encoder(spec, x_mask)

        # content encoder
        mu_y, x_mask, *_ = self.enc_p(
            c,
            x_mask,
            cond=cond,
            cond_mask=cond_mask,
            f0=f0,
            uv=uv,
            ppgs=ppgs,
            utt_emb=g,
        )

        z = (
            torch.randn(
                size=(mu_y.shape[0], self.spec_channels, mu_y.shape[2]),
                device=mu_y.device,
            )
            * temperature
        )
        decoder_outputs = self.decoder.inference(
            z,
            x_mask,
            mu_y,
            n_timesteps,
            spk=g,
            cond=cond,
            cond_mask=cond_mask,
            solver="euler",
        )

        return decoder_outputs, None

    @torch.no_grad()
    def vc(
        self,
        c,
        cond,
        cond_mask,
        f0,
        uv,
        ppgs,
        g=None,
        n_timesteps=10,
        temperature=1.0,
        guidance_scale=0.0,
        solver="euler",
    ):
        # if self.n_speakers > 1 and self.speaker_embedding:
        #     g = F.normalize(g).unsqueeze(-1)

        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)

        # x mask
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(
            c.dtype
        )

        # text encoder
        mu_y, x_mask = self.enc_p.vc(
            c,
            x_mask,
            c_lengths,
            cond=cond,
            cond_mask=cond_mask,
            f0=f0,
            uv=uv,
            ppgs=ppgs,
            utt_emb=g.squeeze(-1),
        )

        # fix length compatibility
        y_max_length = int(c_lengths.max())
        y_max_length_ = commons.fix_len_compatibility(y_max_length)
        mu_y = commons.fix_y_by_max_length(mu_y, y_max_length_)
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, y_max_length_), 1).to(
            c.dtype
        )

        z = torch.randn_like(mu_y) * temperature
        decoder_outputs = self.decoder.inference(
            z,
            x_mask,
            mu_y,
            n_timesteps,
            spk=g.squeeze(-1),
            cond=cond,
            cond_mask=cond_mask,
            solver=solver,
        )
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return decoder_outputs, None

    @torch.no_grad()
    def compute_conditional_latent(self, mels, mel_lengths=None):
        speaker_embeddings = []
        cond_latents = []
        for mel, length in zip(mels, mel_lengths):
            x_mask = torch.unsqueeze(commons.sequence_mask(length, mel.size(2)), 1).to(
                mel.dtype
            )

            # reference mel encoder and perceiver latents
            speaker_embedding, cond_latent, conds_mask = self.mel_encoder(mel, x_mask)
            speaker_embeddings.append(speaker_embedding.squeeze(0))
            cond_latents.append(cond_latent.squeeze(0))

        cond_latents = torch.stack(cond_latents, dim=0)
        speaker_embeddings = torch.stack(speaker_embeddings, dim=0)

        # mean pooling for cond_latents and speaker_embeddings
        speaker_embeddings = speaker_embeddings.mean(dim=0, keepdim=True)
        conds = cond_latents.mean(dim=0, keepdim=True)

        return conds, conds_mask, speaker_embeddings
