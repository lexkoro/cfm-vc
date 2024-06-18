import math
import random

import monotonic_align
import torch
from torch import nn

import modules.commons as commons
from modules.cfm.cfm_neuralode import ConditionalFlowMatching
from modules.content_encoder import ContentEncoder
from modules.duration_predictor import StochasticDurationPredictor
from modules.ppg_decoder import PPGDecoder
from modules.reference_encoder import MelStyleEncoder
from modules.reversal_classifer import SpeakerClassifier


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
        dim_head,
        n_layers,
        kernel_size,
        p_dropout,
        speaker_embedding,
        n_speakers,
        ssl_dim,
        **kwargs,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.dim_head = dim_head
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.speaker_embedding = speaker_embedding
        self.n_speakers = n_speakers
        self.ssl_dim = ssl_dim
        self.out_size = int(4 * (22050 // 256))
        self.train_dp = True

        # content encoder
        self.enc_p = ContentEncoder(
            hidden_channels=hidden_channels,
            ssl_dim=ssl_dim,
            filter_channels=filter_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            utt_emb_dim=speaker_embedding,
        )

        # ppg decoder
        self.ppg_decoder = PPGDecoder(
            n_vocab=40,
            n_feats=spec_channels,
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            n_layers=6,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            utt_emb_dim=speaker_embedding,
        )

        # speaker classifier
        self.speaker_classifier = SpeakerClassifier(
            in_channels=hidden_channels, hidden_channels=512, n_speakers=n_speakers
        )

        # reference mel encoder
        self.mel_encoder = MelStyleEncoder(
            in_channels=spec_channels,
            hidden_channels=hidden_channels,
            utt_channels=speaker_embedding,
            kernel_size=5,
            p_dropout=p_dropout,
            n_heads=n_heads,
            dim_head=dim_head,
        )

        # conditional flow matching decoder
        self.decoder = ConditionalFlowMatching(
            in_channels=spec_channels,
            hidden_channels=hidden_channels,
            out_channels=spec_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            spk_emb_dim=speaker_embedding,
            estimator="dit",
        )

        if self.train_dp:
            # duration predictor
            self.duration_predictor = StochasticDurationPredictor(
                in_channels=hidden_channels,
                filter_channels=hidden_channels,
                kernel_size=3,
                p_dropout=0.5,
                utt_emb_dim=speaker_embedding,
            )

    def forward(self, c, f0, uv, energy, spec, c_lengths, ppg, ppg_lengths, ppg_dur):
        # if self.n_speakers > 1 and self.speaker_embedding:
        #     g = F.normalize(g).unsqueeze(-1)

        # y_mask
        y_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(
            c.dtype
        )
        y_max_length = spec.shape[-1]

        # reference mel encoder
        g, cond, cond_mask = self.mel_encoder(spec, y_mask)

        # content encoder
        x_speaker_classifier, encoded_c, f0_pred, lf0, energy_pred = self.enc_p(
            c,
            y_mask,
            cond=cond,
            cond_mask=cond_mask,
            f0=f0,
            uv=uv,
            energy=energy,
            utt_emb=g,
        )

        # speaker classifier
        speaker_logits = self.speaker_classifier(x_speaker_classifier, y_mask)

        # decode ppgs
        x_p, mu_x, x_mask = self.ppg_decoder(
            x=ppg,
            x_lengths=ppg_lengths,
            cond=encoded_c,
            cond_mask=y_mask,
            speaker_embedding=g,
        )

        # attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.spec_channels
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), spec**2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), spec)
            mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = monotonic_align.maximum_path(
                log_prior,
                attn_mask.squeeze(1),
            ).detach()

        w = torch.sum(attn.unsqueeze(1), -1)

        # duration predictor
        if self.train_dp:
            l_length_sdp = self.duration_predictor(
                x=x_p, x_mask=x_mask, speaker_embedding=g, w=w
            )
            l_length_sdp = l_length_sdp / torch.sum(x_mask)
            loss_dur = torch.sum(l_length_sdp.float())
        else:
            loss_dur = torch.tensor(0.0).to(x_p.device)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(self.out_size, type(None)):
            out_size = min(self.out_size, y_max_length)
            max_offset = (c_lengths - out_size).clamp(0)
            offset_ranges = list(
                zip([0] * max_offset.shape[0], max_offset.cpu().numpy())
            )
            out_offset = torch.LongTensor(
                [
                    torch.tensor(random.choice(range(start, end)) if end > start else 0)
                    for start, end in offset_ranges
                ]
            ).to(c_lengths)

            attn_cut = torch.zeros(
                attn.shape[0],
                attn.shape[1],
                out_size,
                dtype=attn.dtype,
                device=attn.device,
            )
            y_cut = torch.zeros(
                spec.shape[0],
                self.spec_channels,
                out_size,
                dtype=spec.dtype,
                device=spec.device,
            )
            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(spec, out_offset)):
                y_cut_length = out_size + (c_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = commons.sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn = attn_cut  # attn -> [B, text_length, cut_length]. The new alignment path does not begin from top left corner
            spec = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(
            attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)
        ).transpose(1, 2)

        # Compute loss of score-based decoder
        diff_loss, _ = self.decoder.forward(spec, None, y_mask, mu_y, spk=g)

        prior_loss = torch.sum(
            0.5 * ((spec - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask
        )
        prior_loss = prior_loss / (torch.sum(y_mask) * self.spec_channels)

        return (
            prior_loss,
            diff_loss,
            loss_dur,
            f0_pred,
            lf0,
            energy_pred,
            speaker_logits,
        )

    @torch.no_grad()
    def infer(
        self,
        c,
        spec,
        f0,
        uv,
        energy,
        ppg,
        ppg_lengths,
        ppg_dur,
        length_scale=1,
        noise_scale_w=0.8,
        n_timesteps=10,
        temperature=1.0,
        guidance_scale=0.0,
    ):
        # if self.n_speakers > 1 and self.speaker_embedding:
        #     g = F.normalize(g).unsqueeze(-1)
        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)

        # x mask
        y_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(
            c.dtype
        )

        # reference mel encoder
        g, cond, cond_mask = self.mel_encoder(spec, y_mask)

        # content encoder
        _, encoded_c, *_ = self.enc_p(
            c,
            y_mask,
            cond=cond,
            cond_mask=cond_mask,
            f0=f0,
            uv=uv,
            energy=energy,
            utt_emb=g,
        )

        # decode ppgs
        x_p, mu_x, x_mask = self.ppg_decoder(
            x=ppg,
            x_lengths=ppg_lengths,
            cond=encoded_c,
            cond_mask=y_mask,
            speaker_embedding=g,
        )

        # duration predictor
        if self.train_dp:
            logw = self.duration_predictor(
                x=x_p,
                x_mask=x_mask,
                speaker_embedding=g,
                reverse=True,
                noise_scale=noise_scale_w,
            )
            w = torch.exp(logw) * x_mask * length_scale
            w_ceil = torch.ceil(w)
        else:
            w_ceil = ppg_dur.unsqueeze(1)

        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = (
            commons.sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask.dtype)
        )

        # attn
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = commons.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(
            1
        )

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(
            attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)
        ).transpose(1, 2)

        z = (
            torch.randn(
                size=(mu_y.shape[0], self.spec_channels, mu_y.shape[2]),
                device=mu_y.device,
            )
            * temperature
        )
        decoder_outputs = self.decoder.inference(
            z,
            y_mask,
            mu_y,
            n_timesteps,
            spk=g,
            solver="euler",
            guidance_scale=guidance_scale,
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
        energy,
        ppg,
        ppg_lengths,
        ppg_dur,
        g,
        length_scale=1,
        noise_scale_w=0.8,
        n_timesteps=10,
        temperature=1.0,
        guidance_scale=0.0,
        solver="euler",
    ):
        # if self.n_speakers > 1 and self.speaker_embedding:
        #     g = F.normalize(g).unsqueeze(-1)

        c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)

        # x mask
        y_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(
            c.dtype
        )

        # text encoder
        encoded_c = self.enc_p.vc(
            c,
            y_mask,
            cond=cond,
            cond_mask=cond_mask,
            f0=f0,
            uv=uv,
            energy=energy,
            utt_emb=g,
        )

        # decode ppgs
        x_p, mu_x, x_mask = self.ppg_decoder(
            x=ppg,
            x_lengths=ppg_lengths,
            cond=encoded_c,
            cond_mask=y_mask,
            speaker_embedding=g,
        )

        if self.train_dp:
            # duration predictor
            logw = self.duration_predictor(
                x=x_p,
                x_mask=x_mask,
                speaker_embedding=g,
                reverse=True,
                noise_scale=noise_scale_w,
            )
            w = torch.exp(logw) * x_mask * length_scale
            w_ceil = torch.ceil(w)
        else:
            w_ceil = ppg_dur.unsqueeze(1)

        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = (
            commons.sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask.dtype)
        )

        # attn
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = commons.generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(
            1
        )

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(
            attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)
        ).transpose(1, 2)

        z = torch.randn_like(mu_y) * temperature
        decoder_outputs = self.decoder.inference(
            z,
            y_mask,
            mu_y,
            n_timesteps,
            spk=g,
            solver=solver,
            guidance_scale=guidance_scale,
        )
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return decoder_outputs, None

    @torch.no_grad()
    def compute_conditional_latent(self, mels, mel_lengths):
        speaker_embeddings = []
        latents_embeddings = []
        for mel, length in zip(mels, mel_lengths):
            x_mask = torch.unsqueeze(commons.sequence_mask(length, mel.size(2)), 1).to(
                mel.dtype
            )

            # reference mel encoder and perceiver latents
            speaker_embedding, cond, cond_mask = self.mel_encoder(mel, x_mask)
            speaker_embeddings.append(speaker_embedding.squeeze(0))
            latents_embeddings.append(cond.squeeze(0))

        speaker_embedding = torch.stack(speaker_embeddings, dim=0)
        latents_embedding = torch.stack(latents_embeddings, dim=0)

        # mean pooling for cond_latents and speaker_embeddings
        speaker_embedding = speaker_embedding.mean(dim=0, keepdim=True)
        latents_embedding = latents_embedding.mean(dim=0, keepdim=True)

        return speaker_embedding, latents_embedding, cond_mask
