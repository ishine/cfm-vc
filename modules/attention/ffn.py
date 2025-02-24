from typing import Callable

from torch import nn
from torch.nn import functional as F

from modules import commons
from modules.modules import FiLM


class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation: Callable, kernel_size=1):
        super().__init__()
        self.act = activation
        self.proj = nn.Conv1d(dim_in, dim_out * 2, kernel_size)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=1)
        return x * self.act(gate)


class FFN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size=3,
        p_dropout=0.0,
        use_glu=False,
        causal=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = nn.GELU()
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        if use_glu:
            self.conv_1 = GLU(
                in_channels, filter_channels, self.activation, kernel_size=kernel_size
            )
        else:
            self.conv_1 = nn.Sequential(
                nn.Conv1d(in_channels, filter_channels, 1), self.activation
            )

        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x


class FiLMFFN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout=0.0,
        causal=False,
        cond_emb_dim=192,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = nn.GELU()
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.film = FiLM(out_channels, cond_emb_dim)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask, cond):
        x = self.conv_1(self.padding(x * x_mask))
        x = self.activation(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        x = self.film(x, cond)
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x
