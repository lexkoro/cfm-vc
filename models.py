import math

import torch
from torch import nn

import modules.commons as commons
from modules.cfm.cfm_neuralode import ConditionalFlowMatching
from modules.content_encoder import ContentEncoder
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

        # content encoder
        self.enc_p = ContentEncoder(
            hidden_channels=hidden_channels,
            n_feats=spec_channels,
            ssl_dim=ssl_dim,
            filter_channels=filter_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            n_layers=n_layers,
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
            cond_channels=hidden_channels,
            kernel_size=5,
            p_dropout=0.1,
            n_heads=n_heads,
            dim_head=dim_head,
        )

        # conditional flow matching decoder
        self.decoder = ConditionalFlowMatching(
            in_channels=spec_channels,
            hidden_channels=hidden_channels,
            out_channels=spec_channels,
            spk_emb_dim=speaker_embedding,
            estimator="dit",
        )

    def forward(self, c, f0, uv, energy, spec, c_lengths=None):
        # if self.n_speakers > 1 and self.speaker_embedding:
        #     g = F.normalize(g).unsqueeze(-1)

        # x_mask
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(
            c.dtype
        )

        # reference mel encoder
        g, cond, cond_mask = self.mel_encoder(spec, x_mask)

        # content encoder
        x_speaker_classifier, mu_y, x_mask, f0_pred, lf0, energy_pred = self.enc_p(
            c,
            x_mask,
            cond=cond,
            cond_mask=cond_mask,
            f0=f0,
            uv=uv,
            energy=energy,
            utt_emb=g,
        )

        # speaker classifier
        speaker_logits = self.speaker_classifier(x_speaker_classifier, x_mask)

        # Compute loss of score-based decoder
        diff_loss, _ = self.decoder.forward(spec, None, x_mask, mu_y, spk=g)

        prior_loss = torch.sum(
            0.5 * ((spec - mu_y) ** 2 + math.log(2 * math.pi)) * x_mask
        )
        prior_loss = prior_loss / (torch.sum(x_mask) * self.spec_channels)

        return (prior_loss, diff_loss, f0_pred, lf0, energy_pred, speaker_logits)

    @torch.no_grad()
    def infer(
        self,
        c,
        spec,
        f0,
        uv,
        energy,
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
        _, mu_y, x_mask, *_ = self.enc_p(
            c,
            x_mask,
            cond=cond,
            cond_mask=cond_mask,
            f0=f0,
            uv=uv,
            energy=energy,
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
            cond=cond,
            cond_mask=cond_mask,
            f0=f0,
            uv=uv,
            energy=energy,
            utt_emb=g,
        )

        # fix length compatibility
        y_max_length = int(c_lengths.max())

        z = torch.randn_like(mu_y) * temperature
        decoder_outputs = self.decoder.inference(
            z,
            x_mask,
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
