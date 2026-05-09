import torch
from torch import nn

import modules.commons as commons
from modules.cfm.flow_matching import ConditionalFlowMatching
from modules.content_encoder import ContentEncoder
from modules.perceiver_encoder import PerceiverResampler
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
        self.content_encoder = ContentEncoder(
            hidden_channels=hidden_channels,
            ssl_dim=ssl_dim,
            filter_channels=filter_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
        )

        # Speaker Encoder
        self.speaker_encoder = PerceiverResampler(
            in_channels=spec_channels,
            hidden_channels=hidden_channels,
            n_layers=2,
            num_latents=32,
            dim_head=dim_head,
            n_heads=n_heads,
            ff_mult=4,
            p_dropout=p_dropout,
        )

        # spec decoder
        self.decoder = ConditionalFlowMatching(
            estimator_params={
                "in_channels": spec_channels + hidden_channels,
                "hidden_channels": hidden_channels,
                "out_channels": spec_channels,
                "filter_channels": filter_channels,
                "dropout": 0.05,
                "n_layers": 8,
                "n_heads": n_heads,
                "dim_head": dim_head,
                "kernel_size": kernel_size,
            },
        )

    def forward(self, c, f0, uv, spec, c_lengths=None):
        # if self.n_speakers > 1 and self.speaker_embedding:
        #     g = F.normalize(g).unsqueeze(-1)

        # x_mask
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(
            c.dtype
        )

        # speaker encoder
        speaker_cond, speaker_cond_mask = self.speaker_encoder(spec, x_mask)

        # content encoder
        x_speaker_classifier, mu_y, f0_pred, lf0 = self.content_encoder(
            c,
            x_mask,
            cond=speaker_cond,
            cond_mask=speaker_cond_mask,
            f0=f0,
            uv=uv,
        )

        # # speaker classifier
        # speaker_logits = self.speaker_classifier(x_speaker_classifier, x_mask)

        # Compute loss of score-based decoder
        diff_loss, estimator_pred = self.decoder.forward(
            spec,
            x_mask,
            mu_y,
            spk=None,
            cond=speaker_cond,
            cond_mask=speaker_cond_mask,
        )

        return (diff_loss, f0_pred, lf0)

    @torch.no_grad()
    def infer(
        self,
        c,
        spec,
        f0,
        uv,
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

        # speaker encoder
        speaker_cond, speaker_cond_mask = self.speaker_encoder(spec, x_mask)

        # content encoder
        x_speaker_classifier, mu_y, *_ = self.content_encoder(
            c,
            x_mask,
            cond=speaker_cond,
            cond_mask=speaker_cond_mask,
            f0=f0,
            uv=uv,
        )

        z = (
            torch.randn(
                size=(mu_y.shape[0], self.spec_channels, mu_y.shape[2]),
                dtype=mu_y.dtype,
                device=mu_y.device,
            )
            * temperature
        )

        decoder_outputs = self.decoder.inference(
            z,
            x_mask,
            mu_y,
            n_timesteps,
            spk=None,
            cond=speaker_cond,
            cond_mask=speaker_cond_mask,
            guidance_scale=guidance_scale,
            solver=solver,
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

        # content encoder
        mu_y = self.content_encoder.vc(
            c,
            x_mask,
            cond=cond,
            cond_mask=cond_mask,
            f0=f0,
            uv=uv,
        )

        z = (
            torch.randn(
                size=(mu_y.shape[0], self.spec_channels, mu_y.shape[2]),
                dtype=mu_y.dtype,
                device=mu_y.device,
            )
            * temperature
        )

        decoder_outputs = self.decoder.inference(
            z,
            x_mask,
            mu_y,
            n_timesteps,
            spk=None,
            cond=cond,
            cond_mask=cond_mask,
            guidance_scale=guidance_scale,
            solver=solver,
        )

        return decoder_outputs, None

    @torch.no_grad()
    def compute_conditional_latent(self, mels, mel_lengths):
        latents_embeddings = []
        for mel, length in zip(mels, mel_lengths):
            x_mask = torch.unsqueeze(commons.sequence_mask(length, mel.size(2)), 1).to(
                mel.dtype
            )

            # reference mel encoder and perceiver latents
            # speaker encoder
            speaker_cond, speaker_cond_mask = self.speaker_encoder(mel, x_mask)
            latents_embeddings.append(speaker_cond.squeeze(0))

        latents_embedding = torch.stack(latents_embeddings, dim=0)

        # mean pooling for cond_latents and speaker_embeddings
        latents_embedding = latents_embedding.mean(dim=0, keepdim=True)

        return latents_embedding, speaker_cond_mask
