import torch
from torch import nn

from modules.cfm.flow_matching import ConditionalFlowMatching
from modules.commons import rand_span_mask, sequence_mask
from modules.content_encoder import ContentEncoder


def _get(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        value = config.get(key, default)
    else:
        value = getattr(config, key, default)
    return default if value is None else value


class GameVC(nn.Module):
    """
    Game Voice Conversion Model
    """

    def __init__(
        self,
        spec_channels,
        unit_vocab_size=500,
        encoder=None,
        decoder=None,
        hidden_channels=None,
        filter_channels=None,
        n_heads=None,
        dim_head=None,
        n_layers=None,
        kernel_size=None,
        p_dropout=None,
        **kwargs,
    ):
        super().__init__()
        self.spec_channels = spec_channels

        self.encoder_hidden_channels = _get(
            encoder,
            "hidden_channels",
            _get(encoder, "hidden_dim", hidden_channels or 256),
        )
        self.encoder_filter_channels = _get(
            encoder, "filter_channels", filter_channels or 1024
        )
        self.encoder_n_heads = _get(encoder, "n_heads", n_heads or 4)
        self.encoder_dim_head = _get(encoder, "dim_head", dim_head)
        self.encoder_n_layers = _get(encoder, "n_layers", n_layers or 6)
        self.encoder_kernel_size = _get(
            encoder, "kernel_size", _get(encoder, "conv_kernel_size", kernel_size or 15)
        )
        self.encoder_p_dropout = _get(encoder, "p_dropout", p_dropout or 0.1)

        self.decoder_hidden_channels = _get(
            decoder,
            "hidden_channels",
            _get(
                decoder, "hidden_dim", hidden_channels or self.encoder_hidden_channels
            ),
        )
        self.decoder_filter_channels = _get(
            decoder, "filter_channels", filter_channels or self.encoder_filter_channels
        )
        self.decoder_n_heads = _get(decoder, "n_heads", n_heads or self.encoder_n_heads)
        self.decoder_dim_head = _get(decoder, "dim_head", self.encoder_dim_head)
        self.decoder_n_layers = _get(
            decoder, "n_layers", n_layers or self.encoder_n_layers
        )
        self.decoder_kernel_size = _get(
            decoder,
            "kernel_size",
            _get(decoder, "conv_kernel_size", kernel_size or self.encoder_kernel_size),
        )
        self.decoder_p_dropout = _get(
            decoder,
            "p_dropout",
            _get(decoder, "dropout", p_dropout or 0.05),
        )

        # content encoder
        self.content_encoder = ContentEncoder(
            num_units=unit_vocab_size,
            hidden_channels=self.encoder_hidden_channels,
            filter_channels=self.encoder_filter_channels,
            n_heads=self.encoder_n_heads,
            dim_head=self.encoder_dim_head,
            n_layers=self.encoder_n_layers,
            kernel_size=self.encoder_kernel_size,
            p_dropout=self.encoder_p_dropout,
        )

        # spec decoder
        self.decoder = ConditionalFlowMatching(
            in_channels=self.encoder_hidden_channels + (2 * self.spec_channels),
            hidden_channels=self.decoder_hidden_channels,
            filter_channels=self.decoder_filter_channels,
            out_channels=self.spec_channels,
            n_layers=self.decoder_n_layers,
            n_heads=self.decoder_n_heads,
            dim_head=self.decoder_dim_head,
            kernel_size=self.decoder_kernel_size,
            p_dropout=self.decoder_p_dropout,
            use_skip_connections=False,
        )

    def forward(self, units, units_lengths, mel, mel_lengths):
        # Mask
        x_mask = sequence_mask(units_lengths, units.size(1)).unsqueeze(1).to(mel.dtype)

        # Encoder
        unit_features = self.content_encoder(units, x_mask)

        # Build a prefix prompt mask so training matches inference conditioning.
        prompt_mask = rand_span_mask(
            mel, mel_lengths, frac_lengths=(0.1, 0.4), from_start=True
        )

        # Decoder
        diff_loss, estimator_pred, generation_mask = self.decoder.forward(
            x1=mel,
            mask=x_mask,
            mu=unit_features,
            prompt_mask=prompt_mask,
        )
        mel_generated = mel * generation_mask.to(mel.dtype)

        return diff_loss, estimator_pred, mel_generated

    @torch.no_grad()
    def infer(
        self,
        source_units,
        target_units,
        target_mel,
        source_lengths,
        target_lengths,
        n_timesteps=10,
        temperature=1.0,
        guidance_scale=0.0,
        solver="euler",
    ):
        source_max_len = int(source_lengths.max().item())
        target_max_len = int(target_lengths.max().item())

        source_units = source_units[:, :source_max_len]
        target_units = target_units[:, :target_max_len]
        target_mel = target_mel[:, :, :target_max_len]

        # concat source and target units, and create corresponding mask
        units = torch.cat([target_units, source_units], dim=-1)

        # combine target and source lengths
        combined_lengths = target_lengths + source_lengths

        x_mask = (
            sequence_mask(combined_lengths, units.size(1))
            .unsqueeze(1)
            .to(target_mel.dtype)
        )

        # Encoder
        unit_features = self.content_encoder(units, x_mask)

        # Decoder
        decoder_outputs = self.decoder.inference(
            mu=unit_features,
            mask=x_mask,
            target_condition=target_mel,
            source_lengths=source_lengths,
            target_lengths=target_lengths,
            temperature=temperature,
            n_timesteps=n_timesteps,
            guidance_scale=guidance_scale,
            solver=solver,
        )

        # Cut the condition part from the output
        start = int(target_lengths[0].item())
        decoder_outputs = decoder_outputs[:, :, start:]

        return decoder_outputs
