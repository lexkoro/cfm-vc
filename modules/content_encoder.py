from torch import nn

from modules.attention.transformer import Encoder
from modules.modules import Wav2Vec2StackedPositionEncoder


class ContentEncoder(nn.Module):
    def __init__(
        self,
        num_units,
        hidden_channels,
        filter_channels,
        n_heads,
        dim_head,
        n_layers,
        kernel_size,
        p_dropout,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers

        # Unit Embeddings
        self.unit_emb = nn.Embedding(
            num_embeddings=num_units, embedding_dim=hidden_channels
        )

        # Prenet
        self.prenet = Wav2Vec2StackedPositionEncoder(
            depth=2,
            dim=hidden_channels,
            kernel_size=15,
            groups=16,
            p_dropout=p_dropout,
        )

        # Encoder
        self.encoder = Encoder(
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            n_layers=n_layers,
            p_dropout=p_dropout,
        )

    def forward(self, x, x_mask):
        # embedding
        x = self.unit_emb(x).transpose(1, 2)

        # Prenet
        x = self.prenet(x, x_mask) + x

        # encoder
        x = self.encoder(x, x_mask)

        return x
