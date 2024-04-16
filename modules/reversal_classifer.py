import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.parametrizations import weight_norm


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


class GradientReversal(torch.nn.Module):
    """Gradient Reversal Layer
        Y. Ganin, V. Lempitsky,
        "Unsupervised Domain Adaptation by Backpropagation",
        in ICML, 2015.
    Forward pass is the identity function
    In the backward pass, upstream gradients are multiplied by -lambda (i.e. gradient are reversed)
    """

    def __init__(self, lambda_reversal=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_reversal

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class MaskedAdaptiveAvgPool1d(nn.Module):
    """AdaptiveAvgPool1d with mask support."""

    def __init__(self):
        super(MaskedAdaptiveAvgPool1d, self).__init__()

    def forward(self, x, x_mask):
        # Sum the mask to get the count of non-zero values
        non_zero_counts = x_mask.sum(dim=-1, keepdim=True)
        non_zero_counts[non_zero_counts == 0] = 1  # To avoid division by zero

        # Compute the sum of x and then divide by the count of non-zero values
        x_sum = x.sum(dim=-1, keepdim=True)
        x_avg = x_sum / non_zero_counts

        return x_avg


class ReversalClassifier(nn.Module):
    """Adversarial classifier with an optional gradient reversal layer.
    Adapted from: https://github.com/Tomiinek/Multilingual_Text_to_Speech/

    Args:
        in_channels (int): Number of input tensor channels.
        out_channels (int): Number of output tensor channels (Number of classes).
        hidden_channels (int): Number of hidden channels.
        gradient_clipping_bound (float): Maximal value of the gradient which flows from this module. Default: 0.25
        scale_factor (float): Scale multiplier of the reversed gradientts. Default: 1.0
        reversal (bool): If True reversal the gradients. Default: True
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.classifier = nn.Sequential(
            GradientReversal(lambda_reversal=1),
            weight_norm(
                nn.Conv1d(in_channels, self.hidden_channels, kernel_size=5, padding=2)
            ),
            nn.ReLU(),
            weight_norm(
                nn.Conv1d(
                    self.hidden_channels, self.hidden_channels, kernel_size=5, padding=2
                )
            ),
            nn.ReLU(),
            weight_norm(
                nn.Conv1d(self.hidden_channels, out_channels, kernel_size=5, padding=2)
            ),
        )

        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, x, target_embeddings):
        # pass through classifier
        outputs = self.classifier(x)  # (B, nb_speakers)
        outputs = torch.mean(outputs, dim=-1)

        loss = self.loss(target_embeddings, outputs)
        return loss

    def loss(self, target_embeddings, predictions):
        # The target for CosineEmbeddingLoss should be 1 if you want the embeddings to be similar
        return self.cosine_loss(
            predictions,
            target_embeddings,
            torch.Tensor(predictions.size(0)).to(target_embeddings).fill_(1.0),
        )
