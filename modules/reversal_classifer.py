from torch import nn
from torch.autograd import Function

from modules.commons import temporal_avg_pooling
from modules.modules import ConvNorm1D, LinearNorm


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
                in_dim=in_channels,
                out_dim=hidden_channels,
                kernel_size=3,
                padding=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
            ConvNorm1D(
                in_dim=hidden_channels,
                out_dim=hidden_channels,
                kernel_size=3,
                padding=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
        )

        self.fc = LinearNorm(
            in_dim=hidden_channels, out_dim=n_speakers, w_init_gain="linear"
        )

    def forward(self, x, x_mask):
        """Forward function of Speaker Classifier:
        x = (B, embed_dim)
        """
        # pass through classifier
        outputs = self.classifier(x)

        # temporal average pooling
        outputs = temporal_avg_pooling(outputs, x_mask)

        # fc layer
        outputs = self.fc(outputs)

        return outputs
