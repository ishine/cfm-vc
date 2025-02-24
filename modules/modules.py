import math
from typing import Literal

import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

LRELU_SLOPE = 0.1

NORM_TYPES = Literal[
    "layernorm",
    "rmsnorm",
    "adarmsnorm",
    "groupnorm",
    "condlayernorm",
    "condgroupnorm",
    "adainnorm",
    "stylecondnorm",
]


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


def get_normalization(
    norm_type: NORM_TYPES = "rmsnorm",
    dim_out: int = 192,
    utt_emb_dim: int = 0,
):
    """
    Returns the appropriate normalization layer based on the norm_type and utt_emb_dim.

    Args:
        norm_type (str): The type of normalization to be applied.
        dim_out (int): The output dimension of the block.
        utt_emb_dim (int): The dimension of the utterance embedding.

    Returns:
        nn.Module: The normalization layer.
    """
    if norm_type == "condlayernorm" and utt_emb_dim != 0:
        return ConditionalLayerNorm(dim_out, utt_emb_dim)
    elif norm_type == "condgroupnorm" and utt_emb_dim != 0:
        return ConditionalGroupNorm(8, dim_out, utt_emb_dim)
    elif norm_type == "adainnorm" and utt_emb_dim != 0:
        return AdaIN1d(dim_out, utt_emb_dim)
    elif norm_type == "stylecondnorm" and utt_emb_dim != 0:
        return StyleAdaptiveLayerNorm(dim_out, utt_emb_dim)
    elif norm_type == "groupnorm":
        return nn.GroupNorm(8, dim_out)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim_out)
    elif norm_type == "adarmsnorm":
        return AdaptiveRMSNorm(dim_out, utt_emb_dim)
    elif norm_type == "layernorm":
        return LayerNorm(dim_out)
    else:
        raise NotImplementedError(
            f"Normalization type '{norm_type}' not implemented or utt_emb_dim is invalid."
        )


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (kernel_size,) = self.kernel_size
        (dilation,) = self.dilation
        (stride,) = self.stride

        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        causal_padded_x = F.pad(x, (self.causal_padding, 0), value=0.0)
        return super().forward(causal_padded_x)


class ConditionalGroupNorm(nn.Module):
    def __init__(self, groups, normalized_shape, context_dim):
        super().__init__()
        self.norm = nn.GroupNorm(groups, normalized_shape, affine=False)
        self.context_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(context_dim, 2 * normalized_shape)
        )
        self.context_mlp[1].weight.data.zero_()
        self.context_mlp[1].bias.data.zero_()

    def forward(self, x, context):
        context = self.context_mlp(context)
        ndims = " 1" * len(x.shape[2:])
        context = rearrange(context, f"b c -> b c{ndims}")

        scale, shift = context.chunk(2, dim=1)
        x = self.norm(x) * (scale + 1.0) + shift
        return x


class Block1D(torch.nn.Module):
    """
    Block1D module represents a 1-dimensional block in a neural network.

    Args:
        dim (int): The input dimension of the block.
        dim_out (int): The output dimension of the block.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        utt_emb_dim (int, optional): The dimension of the utterance embedding. Defaults to 0.
        norm_type (str, optional): The type of normalization to be applied.
            Possible values are "condlayernorm", "condgroupnorm" and "layernorm". Defaults to "layernorm".
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        utt_emb_dim: int = 0,
        norm_type: NORM_TYPES = "rmsnorm",
    ):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            dim, dim_out, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.norm_type = norm_type

        # get normalization layer
        self.norm = get_normalization(norm_type, dim_out, utt_emb_dim)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, utt_emb=None):
        """
        Forward pass of the Block1D module.

        Args:
            x (torch.Tensor): The input tensor.
            mask (torch.Tensor): The mask tensor.
            utt_emb (torch.Tensor, optional): The utterance embedding tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """

        output = self.conv1d(x * mask)
        if utt_emb is not None:
            output = self.norm(output, utt_emb)
        else:
            output = self.norm(output)
        output = self.act(output)
        output = self.dropout(output)
        return output * mask


class ResnetBlock1D(torch.nn.Module):
    """
    Residual block for 1D ResNet.

    Args:
        dim_in (int): Number of input channels.
        dim_out (int): Number of output channels.
        dropout (float, optional): Dropout rate. Default is 0.1.
        utt_emb_dim (int, optional): Dimension of utterance embedding. Default is 0.
        norm_type (str, optional): The type of normalization to be applied.
            Possible values are "condlayernorm", "condgroupnorm" and "layernorm". Defaults to "layernorm".
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        utt_emb_dim: int = 0,
        norm_type: NORM_TYPES = "rmsnorm",
    ):
        super().__init__()

        self.norm_type = norm_type

        self.block1 = Block1D(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            dropout=dropout,
            utt_emb_dim=utt_emb_dim,
            norm_type=norm_type,
        )
        self.block2 = Block1D(
            dim_out,
            dim_out,
            kernel_size=kernel_size,
            dropout=dropout,
            utt_emb_dim=utt_emb_dim,
            norm_type=norm_type,
        )

        self.res_conv = (
            torch.nn.Conv1d(dim_in, dim_out, 1)
            if dim_in != dim_out
            else torch.nn.Identity()
        )

    def forward(self, x, mask, utt_emb=None):
        """
        Forward pass of the ResnetBlock1D.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length).
            mask (torch.Tensor): Mask tensor of shape (batch_size, sequence_length).
            utt_emb (torch.Tensor, optional): Utterance embedding tensor of shape (batch_size, utt_emb_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, sequence_length).
        """
        h = self.block1(x, mask, utt_emb)
        h = self.block2(h, mask, utt_emb)
        output = h + self.res_conv(x * mask)
        return output


class ConvNorm1D(nn.Module):
    """Conv Norm 1D Module:
    - Conv 1D
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm1D, self).__init__()
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
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


class AdaIN1d(nn.Module):
    def __init__(self, num_features, style_dim):
        super().__init__()
        """
        Adaptive Instance Normalization (AdaIN) layer.

        Args:
            num_features (int): Number of features in the condition sequence.
            style_dim (int): Dimension of the speaker embedding.
        """
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.context_mlp = nn.Linear(style_dim, 2 * num_features)

    def forward(self, x, s):
        if s.dim() == 3:
            s = s.squeeze(-1)

        h = self.context_mlp(s).unsqueeze(-1)
        gamma, beta = h.chunk(2, dim=1)

        return (1 + gamma) * self.norm(x) + beta


class ResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size: int = 3,
        actv: nn.Module = nn.LeakyReLU(0.2),
        normalize: bool = False,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.register_buffer("sqrt", torch.sqrt(torch.FloatTensor([0.5])).squeeze(0))
        self.dropout = nn.Dropout(p=dropout_p)

        self.conv1 = weight_norm(
            nn.Conv1d(
                dim_in,
                dim_in,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )
        self.conv2 = weight_norm(
            nn.Conv1d(
                dim_in,
                dim_out,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )

        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)

        self.shortcut = (
            weight_norm(nn.Conv1d(dim_in, dim_out, 1, padding=0, bias=False))
            if dim_in != dim_out
            else nn.Identity()
        )

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.dropout(x)
        x = self.conv1(x)

        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._residual(x) + self.shortcut(x)
        return x * self.sqrt


class AdainResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hidden=None,
        kernel_size=3,
        style_dim=64,
        actv=nn.LeakyReLU(0.2),
        dropout_p=0.0,
    ):
        super().__init__()

        if dim_hidden is None:
            dim_hidden = dim_out
        self.actv = actv
        self.learned_sc = dim_in != dim_out
        self.dropout = nn.Dropout(dropout_p)
        self.register_buffer("sqrt", torch.sqrt(torch.FloatTensor([0.5])).squeeze(0))

        self.conv1 = weight_norm(
            nn.Conv1d(
                dim_in,
                dim_hidden,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )
        self.conv2 = weight_norm(
            nn.Conv1d(
                dim_hidden,
                dim_out,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )
        self.norm1 = AdaIN1d(dim_in, style_dim)
        self.norm2 = AdaIN1d(dim_hidden, style_dim)
        self.shortcut = (
            nn.Conv1d(dim_in, dim_out, 1, padding=0, bias=False)
            if dim_in != dim_out
            else nn.Identity()
        )

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x

    def forward(self, x, s):
        x = self.shortcut(x) + self._residual(x, s)
        return x * self.sqrt


class ConditionalLayerNorm(nn.Module):
    """
    Conditional Layer Normalization module

    Args:
        embedding_dim (int): Dimension of the target embeddings.
        utt_emb_dim (int): Dimension of the speaker embeddings.
    """

    def __init__(self, embedding_dim: int, utt_emb_dim: int, epsilon: float = 1e-6):
        super(ConditionalLayerNorm, self).__init__()

        self.embedding_dim = embedding_dim
        self.utt_emb_dim = utt_emb_dim
        self.epsilon = epsilon

        self.W_scale = nn.Linear(utt_emb_dim, embedding_dim)
        self.W_bias = nn.Linear(utt_emb_dim, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)

    def forward(self, x: torch.Tensor, utt_emb: torch.Tensor):
        x = x.transpose(1, 2)
        if utt_emb.dim() == 3:
            utt_emb = utt_emb.squeeze(-1)

        scale = self.W_scale(utt_emb).unsqueeze(1)
        bias = self.W_bias(utt_emb).unsqueeze(1)

        x = nn.functional.layer_norm(x, (self.embedding_dim,), eps=self.epsilon)
        x = x * scale + bias
        x = x.transpose(1, 2)
        return x


class SwishBlock(nn.Module):
    """Swish Block"""

    def __init__(self, in_channels, hidden_dim, out_channels):
        super(SwishBlock, self).__init__()
        self.layer = nn.Sequential(
            LinearNorm(in_channels, hidden_dim, bias=True),
            nn.SiLU(),
            LinearNorm(hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
            LinearNorm(hidden_dim, out_channels, bias=True),
        )

    def forward(self, S, E, V):
        out = torch.cat(
            [
                S.unsqueeze(-1),
                E.unsqueeze(-1),
                V.unsqueeze(1).expand(-1, E.size(1), -1, -1),
            ],
            dim=-1,
        )
        out = self.layer(out)

        return out


class ConvBlock(nn.Module):
    """Convolutional Block"""

    def __init__(
        self, in_channels, out_channels, kernel_size, dropout, activation=nn.ReLU()
    ):
        super(ConvBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            ConvNorm(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="tanh",
            ),
            LayerNorm(out_channels),
            activation,
        )
        self.dropout = dropout
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, enc_input, mask=None):
        enc_output = enc_input.contiguous().transpose(1, 2)
        enc_output = F.dropout(self.conv_layer(enc_output), self.dropout, self.training)

        enc_output = self.layer_norm(enc_output.contiguous().transpose(1, 2))
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output


class FiLM(nn.Module):
    def __init__(self, in_dim, cond_dim):
        super().__init__()
        """
        FiLM layer.

        Args:
            in_dim (int): Input dimension of the phoneme sequence.
            cond_dim (int): Dimension of the condition sequence.
        """

        self.gain = nn.Conv1d(cond_dim, in_dim, kernel_size=1)
        self.bias = nn.Conv1d(cond_dim, in_dim, kernel_size=1)

        nn.init.xavier_uniform_(self.gain.weight)
        nn.init.constant_(self.gain.bias, 1)

        nn.init.xavier_uniform_(self.bias.weight)
        nn.init.constant_(self.bias.bias, 0)

    def forward(self, x, condition):
        gain = self.gain(condition)
        bias = self.bias(condition)

        return x * (1.0 + gain) + bias


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        use_weight_norm=False,
    ):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_weight_norm = use_weight_norm
        conv_fn = torch.nn.Conv1d
        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )
        if self.use_weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, signal, mask=None):
        conv_signal = self.conv(signal)
        if mask is not None:
            # always re-zero output if mask is
            # available to match zero-padding
            conv_signal = conv_signal * mask
        return conv_signal


class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class BottleneckLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        norm="weightnorm",
        non_linearity="relu",
        kernel_size=3,
    ):
        super(BottleneckLayer, self).__init__()

        self.bottleneck_channels = bottleneck_channels

        self.proj = ConvNorm(
            in_channels,
            bottleneck_channels,
            kernel_size=kernel_size,
            use_weight_norm=(norm == "weightnorm"),
        )

        if norm == "instancenorm":
            self.proj = nn.Sequential(
                self.proj, nn.InstanceNorm1d(bottleneck_channels, affine=True)
            )

        if non_linearity == "leakyrelu":
            self.act = nn.LeakyReLU(0.3)
        elif non_linearity == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.act(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim, unit_offset=False):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim**0.5

        self.g = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.g, 1.0 - float(unit_offset))

    def forward(self, x, *args):
        x = x.transpose(1, -1)
        gamma = self.g + float(self.unit_offset)
        x = F.normalize(x, dim=-1) * self.scale * gamma
        return x.transpose(1, -1)


class AdaptiveRMSNorm(nn.Module):
    def __init__(self, dim, dim_condition=None):
        super().__init__()
        self.scale = dim**0.5
        dim_condition = default(dim_condition, dim)

        self.to_gamma = nn.Linear(dim_condition, dim, bias=False)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, condition):
        x = x.transpose(1, -1)
        if condition.ndim == 2:
            condition = rearrange(condition, "b d -> b 1 d")

        normed = F.normalize(x, dim=-1)
        gamma = self.to_gamma(condition)
        x = normed * self.scale * (gamma + 1.0)
        return x.transpose(1, -1)


class LayerNorm(nn.Module):
    def __init__(self, channels, utt_emb=0, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x, *args):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class StyleAdaptiveLayerNorm(nn.Module):
    def __init__(self, normalized_shape, style_dim, eps=1e-5):
        super().__init__()
        self.in_dim = normalized_shape
        self.style_dim = style_dim
        self.norm = nn.LayerNorm(self.in_dim, eps=eps, elementwise_affine=False)
        self.style = nn.Sequential(
            nn.SiLU(), nn.Linear(style_dim, 2 * normalized_shape)
        )
        self.style[-1].bias.data[: self.in_dim] = 1
        self.style[-1].bias.data[self.in_dim :] = 0

    def forward(self, x, condition):
        # x: (B, d, T); condition: (B, d, T)
        x = x.transpose(1, -1)

        condition = torch.mean(condition, dim=2)
        style = self.style(condition).unsqueeze(1)
        gamma, beta = torch.chunk(style, chunks=2, dim=2)

        out = self.norm(x)

        out = gamma * out + beta

        return out.transpose(1, -1)


class ConvPositionEmbed(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        groups,
        n_layers=1,
        kernel_size=7,
        p_dropout=0.0,
    ):
        super().__init__()

        filter_channels = 2 * out_channels

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            conv_embed = nn.Conv1d(in_channels, out_channels, 1)
            dwconv = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=groups,
            )
            norm = get_normalization("rmsnorm", out_channels)
            pwconv = nn.Sequential(
                nn.Linear(out_channels, filter_channels),
                nn.GELU(),
                nn.Linear(filter_channels, out_channels),
                nn.Dropout(p_dropout),
            )
            self.layers.append(nn.ModuleList([conv_embed, dwconv, norm, pwconv]))

    def forward(self, x, x_mask, *args):
        for conv_embed, dwconv, norm, pwconv in self.layers:
            y = torch.cat([x, *args], dim=1)
            y = conv_embed(y * x_mask)
            y = dwconv(y)
            y = norm(y).transpose(1, 2)
            y = pwconv(y).transpose(1, 2)
            x = x + y
        return x * x_mask


class DDSConv(nn.Module):
    """
    Dialted and Depth-Separable Convolution
    """

    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x, x_mask, g=None):
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


class DepthSeperableConv(nn.Module):
    """
    Depth-Separable Convolution
    """

    def __init__(
        self,
        channels,
        kernel_size,
        n_layers,
        p_dropout=0.0,
        cond_emb_dim=0,
        utt_emb_dim=0,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(n_layers):
            self.convs_sep.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    groups=channels,
                    padding=kernel_size // 2,
                )
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(StyleAdaptiveLayerNorm(channels, cond_emb_dim))
            self.norms_2.append(ConditionalLayerNorm(channels, utt_emb_dim))

    def forward(self, x, x_mask, utt_emb, cond_emb):
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y, cond_emb)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y, utt_emb)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        utt_emb_dim: int = 0,
        p_dropout: float = 0.0,
        use_grn: bool = False,
        norm_type: NORM_TYPES = "rmsnorm",
    ):
        super().__init__()

        self.dwconv = nn.Conv1d(
            in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels
        )

        self.norm = get_normalization(norm_type, in_channels, utt_emb_dim)
        self.pwconv = nn.Sequential(
            nn.Linear(in_channels, filter_channels),
            nn.GELU(),
            GRN(filter_channels) if use_grn else nn.Identity(),
            nn.Linear(filter_channels, in_channels),
            nn.Dropout(p_dropout),
        )

    def forward(self, x, x_mask, g=None) -> torch.Tensor:
        h = self.dwconv(x) * x_mask
        h = self.norm(h, g).transpose(1, 2)
        h = self.pwconv(h).transpose(1, 2)
        x = h + x
        return x * x_mask


class ConvNeXt(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        out_channels: int,
        n_layers: int = 1,
        utt_emb_dim: int = 0,
        p_dropout: float = 0.0,
        use_grn: bool = False,
        norm_type: NORM_TYPES = "rmsnorm",
    ):
        super().__init__()

        # convnext blocks
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(
                ConvNeXtBlock(
                    in_channels=in_channels,
                    filter_channels=filter_channels,
                    utt_emb_dim=utt_emb_dim,
                    p_dropout=p_dropout,
                    use_grn=use_grn,
                    norm_type=norm_type,
                )
            )

        self.conv_res = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, x_mask, utt_emb=None) -> torch.Tensor:
        # convnext blocks
        for layer in self.layers:
            x = layer(x, x_mask, utt_emb)

        x = self.conv_res(x) * x_mask

        return x


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


class ConditionalConv1dGLU(nn.Module):
    """
    Conv1d + GLU(Gated Linear Unit) with residual connection.
    For GLU refer to https://arxiv.org/abs/1612.08083 paper.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        utt_emb_dim: int,
        p_dropout: float,
    ):
        super(ConditionalConv1dGLU, self).__init__()
        self.out_channels = in_channels
        self.hidden_channels = 2 * in_channels

        self.conv = nn.Conv1d(
            in_channels,
            self.hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.utt_projection = nn.Linear(utt_emb_dim, in_channels)
        self.softsign = nn.Softsign()
        self.dropout = nn.Dropout(p_dropout)
        self.register_buffer("sqrt", torch.sqrt(torch.FloatTensor([0.5])).squeeze(0))

    def forward(self, x, utt_emb) -> torch.Tensor:
        residual = x

        # conv1d + glu
        x = self.conv(x)
        x1, gate = x.chunk(2, dim=1)

        # embed utt emb into x
        utt_emb = self.utt_projection(utt_emb)
        utt_emb = repeat(utt_emb, "b d -> b d n", n=x.size(-1))
        x1 = x1 + self.softsign(utt_emb)

        # glu
        x = x1 * torch.sigmoid(gate)

        x = residual + self.dropout(x)
        return x * self.sqrt


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class Conv1dGLU(nn.Module):
    """
    Conv1d + GLU(Gated Linear Unit) with residual connection.
    For GLU refer to https://arxiv.org/abs/1612.08083 paper.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(Conv1dGLU, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(
            in_channels,
            2 * out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = residual + self.dropout(x)
        return x


class Downsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        elif self.layer_type == "timepreserve":
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == "half":
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError(
                "Got unexpected donwsampletype %s, expected is [none, timepreserve, half]"
                % self.layer_type
            )


class Log(nn.Module):
    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class ElementwiseAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


class ConvFlow(nn.Module):
    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        n_layers,
        num_bins=10,
        tail_bound=5.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = nn.Conv1d(
            filter_channels, self.half_channels * (num_bins * 3 - 1), 1
        )
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, cx?, t] -> [b, c, t, ?]

        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(
            self.filter_channels
        )
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        if not reverse:
            return x, logdet
        else:
            return x


class ConvLSTMLinear(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        n_layers=2,
        n_channels=256,
        kernel_size=3,
        p_dropout=0.1,
        lstm_type="bilstm",
        use_linear=True,
        utt_emb_dim=0,
    ):
        super(ConvLSTMLinear, self).__init__()
        self.out_dim = out_dim
        self.lstm_type = lstm_type
        self.use_linear = use_linear
        self.dropout = nn.Dropout(p=p_dropout)
        self.act = nn.SiLU()
        self.utt_emb_dim = utt_emb_dim

        convolutions = []
        self.convolutions = nn.ModuleList(convolutions)
        if self.utt_emb_dim > 0:
            self.norms = torch.nn.ModuleList()

        for i in range(n_layers):
            in_channels = in_dim if i == 0 else n_channels
            conv_layer = ConvNorm(
                in_channels,
                n_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            )
            conv_layer = torch.nn.utils.weight_norm(conv_layer.conv, name="weight")
            self.convolutions += [conv_layer]
            if self.utt_emb_dim > 0:
                self.norms += [
                    ConditionalLayerNorm(
                        embedding_dim=n_channels,
                        utt_emb_dim=utt_emb_dim,
                        epsilon=1e-6,
                    )
                ]

        if not self.use_linear:
            n_channels = out_dim

        if self.lstm_type != "":
            use_bilstm = False
            lstm_channels = n_channels
            if self.lstm_type == "bilstm":
                use_bilstm = True
                lstm_channels = int(n_channels // 2)

            self.bilstm = nn.LSTM(
                n_channels, lstm_channels, 1, batch_first=True, bidirectional=use_bilstm
            )
            lstm_norm_fn_pntr = nn.utils.spectral_norm
            self.bilstm = lstm_norm_fn_pntr(self.bilstm, "weight_hh_l0")
            if self.lstm_type == "bilstm":
                self.bilstm = lstm_norm_fn_pntr(self.bilstm, "weight_hh_l0_reverse")

        if self.use_linear:
            self.dense = nn.Linear(n_channels, out_dim)

    def run_padded_sequence(self, context, lens, utt_embed=None):
        context_embedded = []
        for b_ind in range(context.size()[0]):
            curr_context = context[b_ind : b_ind + 1, :, : lens[b_ind]].clone()
            if self.utt_emb_dim > 0:
                for conv, norm in zip(self.convolutions, self.norms):
                    xs = conv(curr_context)  # (B, C, Tmax)
                    xs = norm(xs, utt_embed)
                    curr_context = self.dropout(self.act(xs))
            else:
                for conv in self.convolutions:
                    curr_context = self.dropout(self.act(conv(curr_context)))
            context_embedded.append(curr_context[0].transpose(0, 1))
        context = torch.nn.utils.rnn.pad_sequence(context_embedded, batch_first=True)
        return context

    def run_unsorted_inputs(self, fn, context, lens):
        lens_sorted, ids_sorted = torch.sort(lens, descending=True)
        unsort_ids = [0] * lens.size(0)
        for i in range(len(ids_sorted)):
            unsort_ids[ids_sorted[i]] = i
        lens_sorted = lens_sorted.long().cpu()

        context = context[ids_sorted]
        context = nn.utils.rnn.pack_padded_sequence(
            context, lens_sorted, batch_first=True
        )
        context = fn(context)[0]
        context = nn.utils.rnn.pad_packed_sequence(context, batch_first=True)[0]

        # map back to original indices
        context = context[unsort_ids]
        return context

    def forward(self, context, lens, utt_embed=None):
        if context.size()[0] > 1:
            context = self.run_padded_sequence(context, lens, utt_embed)
            # to B, D, T
            context = context.transpose(1, 2)
        else:
            if self.utt_emb_dim > 0:
                for conv, norm in zip(self.convolutions, self.norms):
                    xs = conv(context)  # (B, C, Tmax)
                    xs = norm(xs, utt_embed)
                    context = self.dropout(self.act(xs))
            else:
                for conv in self.convolutions:
                    context = self.dropout(self.act(conv(context)))

        if self.lstm_type != "":
            context = context.transpose(1, 2)
            self.bilstm.flatten_parameters()
            if lens is not None:
                context = self.run_unsorted_inputs(self.bilstm, context, lens)
            else:
                context = self.bilstm(context)[0]
            context = context.transpose(1, 2)

        x_hat = context
        if self.use_linear:
            x_hat = self.dense(context.transpose(1, 2)).transpose(1, 2)

        return x_hat


class Wav2Vec2PositionEncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        groups: int,
    ) -> None:
        super().__init__()

    def forward(self, encodings: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        encodings = self.conv(encodings)
        encodings = self.layer_norm(encodings)
        encodings = self.activation(encodings, cond_emb)
        return encodings


class Wav2Vec2StackedPositionEncoder(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        kernel_size: int,
        groups: int,
        cond_emb_dim: int = 0,
        norm_type: NORM_TYPES = "rmsnorm",
    ) -> None:
        super().__init__()

        k = max(3, kernel_size // depth)

        self.layers = nn.ModuleList()

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.Conv1d(
                            dim,
                            dim,
                            kernel_size,
                            padding="same",
                            groups=groups,
                        ),
                        get_normalization(norm_type, dim, cond_emb_dim),
                        nn.GELU(),
                    ]
                )
            )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, cond_emb: torch.Tensor
    ) -> torch.Tensor:
        for conv, norm, activation in self.layers:
            x = conv(x)
            x = norm(x, cond_emb)
            x = activation(x)

        return x * mask
