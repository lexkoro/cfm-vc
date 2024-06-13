import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def temporal_avg_pooling(x, mask):
    len_ = mask.sum(dim=2)
    x = torch.sum(x * mask, dim=2)
    out = torch.div(x, len_)
    return out


class LinearNorm(nn.Module):
    """Linear Norm Module:
    - Linear Layer
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        """Forward function of Linear Norm
        x = (*, in_dim)
        """
        x = self.linear_layer(x)  # (*, out_dim)

        return x


class ConvNorm1D(nn.Module):
    """Conv Norm 1D Module:
    - Conv 1D
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        """Forward function of Conv Norm 1D
        x = (B, L, in_channels)
        """
        x = self.conv(x)  # (B, out_channels, L)

        return x


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    """Gradient Reversal Layer
        Y. Ganin, V. Lempitsky,
        "Unsupervised Domain Adaptation by Backpropagation",
        in ICML, 2015.
    Forward pass is the identity function
    In the backward pass, upstream gradients are multiplied by -lambda (i.e. gradient are reversed)
    """

    def __init__(self):
        super(GradientReversal, self).__init__()
        self.lambda_ = 1.0

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class SpeakerClassifier(nn.Module):
    """Speaker Classifier Module:
    - 3x Linear Layers with ReLU
    """

    def __init__(self, in_channels, hidden_channels, n_speakers):
        super(SpeakerClassifier, self).__init__()

        self.classifier = nn.Sequential(
            GradientReversal(),
            ConvNorm1D(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
            ConvNorm1D(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
        )
        self.fc = LinearNorm(in_dim=hidden_channels, out_dim=n_speakers)

    def forward(self, x, x_mask):
        """Forward function of Speaker Classifier:
        x = (B, embed_dim)
        """

        # pass through classifier
        outputs = self.classifier(x) * x_mask  # (B, nb_speakers)

        # temporal average pooling
        outputs = temporal_avg_pooling(outputs, x_mask)

        # fc
        outputs = self.fc(outputs)

        return outputs
