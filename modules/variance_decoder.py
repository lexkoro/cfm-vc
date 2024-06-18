import torch
from torch import nn

import modules.attentions as attentions
from modules.modules import AdainResBlk1d, ConditionalLayerNorm


class ConditionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, d_model, style_dim=512):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=d_model)
        self.adain = AdainResBlk1d(
            dim_in=d_model, dim_out=d_model, style_dim=style_dim, kernel_size=3
        )

    def forward(self, x, x_mask, utt_emb):
        emb = self.embed(x).transpose(1, 2)
        x = self.adain(emb, utt_emb)
        return x * x_mask


class AuxDecoder(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        kernel_size,
        n_layers,
        n_heads,
        dim_head=None,
        p_dropout=0.0,
        utt_emb_dim=0,
    ):
        super().__init__()

        self.aux_prenet = nn.Conv1d(
            input_channels, hidden_channels, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.prenet = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )

        self.aux_decoder = attentions.Encoder(
            hidden_channels=hidden_channels,
            filter_channels=hidden_channels * 4,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dim_head=dim_head,
            utt_emb_dim=utt_emb_dim,
            p_dropout=p_dropout,
            causal_ffn=True,
        )

        self.proj = nn.Conv1d(hidden_channels, output_channels, 1)

    def forward(self, x, x_mask, aux, cond, cond_mask, utt_emb):
        # detach x
        x = torch.detach(x)

        # prenets
        x = x + self.aux_prenet(aux) * x_mask
        x = self.prenet(x) * x_mask

        # attention
        x = self.aux_decoder(x, x_mask, cond, cond_mask, utt_emb)

        # out projection
        x = self.proj(x) * x_mask

        return x * x_mask


class VarianceDecoder(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        kernel_size=3,
        n_layers=2,
        n_blocks=2,
        p_dropout=0.1,
        utt_emb_dim=0,
    ):
        """
        Initialize variance encoder module.

        Args:
            input_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            output_channels (int): Number of output channels.
            kernel_size (int): Kernel size of convolution layers.
            n_layers (int): Number of layers.
            n_blocks (int): Number of blocks.
            p_dropout (float): Dropout probability.
            utt_emb_dim (int): Dimension of utterance embedding.
        """
        super().__init__()

        # prenet
        layers = []
        for _ in range(n_blocks):
            for _ in range(n_layers):
                layers.append(
                    nn.ModuleList(
                        [
                            torch.nn.Conv1d(
                                input_channels,
                                hidden_channels,
                                kernel_size,
                                padding=(kernel_size - 1) // 2,
                            ),
                            nn.LeakyReLU(0.2),
                            ConditionalLayerNorm(
                                hidden_channels, utt_emb_dim, epsilon=1e-6
                            ),
                            nn.Dropout(p_dropout),
                        ]
                    )
                )
                input_channels = hidden_channels

            layers.append(
                nn.GRU(
                    hidden_channels,
                    hidden_channels,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )

        self.layers = nn.ModuleList(layers)

        self.proj = nn.Sequential(
            nn.Conv1d(hidden_channels, output_channels, 1),
            nn.InstanceNorm1d(output_channels, affine=True),
        )

    def forward(self, x, x_mask=None, utt_emb=None):
        # attention mask
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)

        for layer in self.layers:
            if isinstance(layer, attentions.MultiHeadAttention):
                x = layer(x, x, attn_mask=attn_mask) + x
            else:
                conv, act, norm, drop = layer
                x = conv(x) * x_mask  # (B, C, Tmax)
                x = act(x)
                x = norm(x, utt_emb)
                x = drop(x)

        x = self.proj(x * x_mask)
        return x * x_mask
