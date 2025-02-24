import math
from typing import Optional

import einx
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat

from modules.attention.transformer import (
    FFN,
    ConditionalMultiHeadAttention,
    TransformerBlock,
)
from modules.modules import GRN, NORM_TYPES, Block1D, get_normalization


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ResnetBlock1D(torch.nn.Module):
    """
    Residual block for 1D ResNet.

    Args:
        dim_in (int): Number of input channels.
        dim_out (int): Number of output channels.
        dropout (float, optional): Dropout rate. Default is 0.1.
        utt_emb_dim (int, optional): Dimension of utterance embedding. Default is 512.
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
        time_emb_dim: int = 512,
        norm_type: NORM_TYPES = "rmsnorm",
    ):
        super().__init__()

        self.norm_type = norm_type

        self.mlp = torch.nn.Sequential(
            nn.SiLU(), torch.nn.Linear(time_emb_dim, dim_out)
        )

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

    def forward(self, x, mask, t, utt_emb=None):
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
        h += self.mlp(t).unsqueeze(-1)
        h = self.block2(h, mask, utt_emb)
        output = h + self.res_conv(x * mask)
        return output


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_channels,
        use_grn=True,
        p_dropout=0.0,
        utt_emb_dim=0,
        norm_type: NORM_TYPES = "rmsnorm",
    ):
        super().__init__()

        # prenet
        self.pre = (
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.dwconv = nn.Conv1d(
            out_channels, out_channels, kernel_size=7, padding=3, groups=out_channels
        )

        self.norm = get_normalization(norm_type, out_channels, utt_emb_dim)

        # pointwise conv
        self.pwconv = nn.Sequential(
            nn.Linear(out_channels, filter_channels),
            nn.GELU(),
            GRN(filter_channels) if use_grn else nn.Identity(),
            nn.Linear(filter_channels, out_channels),
        )

        self.conv_res = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, x_mask, c=None) -> torch.Tensor:
        h = self.pre(x) * x_mask
        h = self.dwconv(h) * x_mask
        h = self.norm(h, c)
        h = rearrange(h, "b d n -> b n d")
        h = self.pwconv(h)
        h = rearrange(h, "b n d -> b d n")
        x = self.dropout(h) + self.conv_res(x)
        return x * x_mask


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        dim_head: Optional[int] = None,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        time_emb_dim: int = 0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.kernel_size = kernel_size

        # self attention
        self.norm_1 = get_normalization("adarmsnorm", hidden_channels, time_emb_dim)
        self.norm_registers = get_normalization(
            "adarmsnorm", hidden_channels, time_emb_dim
        )
        self.attn = ConditionalMultiHeadAttention(
            hidden_channels,
            hidden_channels,
            n_heads,
            dim_head=dim_head,
            p_dropout=p_dropout,
        )

        # feed forward
        self.norm_2 = get_normalization("adarmsnorm", hidden_channels, time_emb_dim)
        self.ffn = FFN(
            hidden_channels,
            hidden_channels,
            filter_channels,
            p_dropout=p_dropout,
            use_glu=True,
        )

    def forward(self, x, x_mask, registers, attn_mask, t):
        # apply mask
        x = x * x_mask

        # self-attention
        attn_input = self.norm_1(x, t)
        registers = self.norm_registers(registers, t)
        x = self.attn(attn_input, c=registers, attn_mask=attn_mask) + x

        # feed-forward
        ff_input = self.norm_2(x, t)

        x = self.ffn(ff_input, x_mask) + x

        return x * x_mask


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# random projection fourier embedding


class RandomFourierEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("weights", torch.randn(dim // 2))

    def forward(self, x):
        freqs = einx.multiply("i, j -> i j", x, self.weights) * 2 * torch.pi
        fourier_embed, _ = pack((x, freqs.sin(), freqs.cos()), "b *")
        return fourier_embed


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_channels, filter_channels=256):
        super().__init__()

        self.layer = nn.Sequential(
            RandomFourierEmbed(hidden_channels),
            nn.Linear(hidden_channels + 1, filter_channels),
            nn.SiLU(),
            nn.Linear(filter_channels, hidden_channels),
        )

    def forward(self, x):
        return self.layer(x)


class Base2FourierFeatures(nn.Module):
    def __init__(self, start=0, stop=8, step=1):
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def __call__(self, inputs):
        freqs = range(self.start, self.stop, self.step)

        # Create Base 2 Fourier features
        w = (
            2.0 ** (torch.tensor(freqs, dtype=inputs.dtype)).to(inputs.device)
            * 2
            * torch.pi
        )
        w = torch.tile(w[None, :, None], (1, inputs.shape[1], 1))

        # Compute features
        h = torch.repeat_interleave(inputs, len(freqs), dim=1)
        h = w * h
        h = torch.stack([torch.sin(h), torch.cos(h)], dim=2)
        return h.reshape(h.size(0), -1, h.size(3))


class CrossCondition(nn.Module):
    def __init__(self, dim, dim_text, cond_audio_to_text=True):
        super().__init__()
        self.text_to_audio = nn.Conv1d(dim_text + dim, dim, 1, bias=False)
        nn.init.zeros_(self.text_to_audio.weight)

        self.cond_audio_to_text = cond_audio_to_text

        if cond_audio_to_text:
            self.audio_to_text = nn.Conv1d(dim + dim_text, dim_text, 1, bias=False)
            nn.init.zeros_(self.audio_to_text.weight)

    def forward(self, audio, text):
        audio_text, _ = pack((audio, text), "b * n")

        text_cond = self.text_to_audio(audio_text)
        audio_cond = self.audio_to_text(audio_text) if self.cond_audio_to_text else 0.0

        return audio + text_cond, text + audio_cond


# reference: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/decoder.py
class DiT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        filter_channels: int,
        n_layers: int = 1,
        n_heads: int = 4,
        dim_head=None,
        num_registers: int = 32,
        kernel_size: int = 3,
        p_dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.n_layers = n_layers

        # prenet

        self.prenet = ResnetBlock1D(
            dim_in=in_channels,
            dim_out=hidden_channels,
            kernel_size=kernel_size,
            dropout=p_dropout,
            time_emb_dim=hidden_channels,
            norm_type="rmsnorm",
        )

        # time embeddings

        self.time_mlp = TimestepEmbedding(hidden_channels, filter_channels)

        # registers

        self.num_registers = num_registers
        self.registers = nn.Parameter(torch.zeros(hidden_channels, num_registers))
        nn.init.normal_(self.registers, std=0.02)

        # blocks

        self.blocks = nn.ModuleList()
        for idx in range(n_layers):
            is_last = idx == n_layers - 1
            self.blocks.append(
                nn.ModuleList(
                    [
                        TransformerBlock(
                            hidden_channels=hidden_channels,
                            filter_channels=filter_channels,
                            n_heads=n_heads,
                            dim_head=dim_head,
                            p_dropout=p_dropout,
                            cond_emb_dim=hidden_channels,
                            norm_type="adarmsnorm",
                        ),
                        CrossCondition(
                            hidden_channels,
                            hidden_channels,
                            cond_audio_to_text=not is_last,
                        ),
                        Encoder(
                            hidden_channels=hidden_channels,
                            filter_channels=filter_channels,
                            n_heads=n_heads,
                            dim_head=dim_head,
                            kernel_size=kernel_size,
                            p_dropout=p_dropout,
                            time_emb_dim=hidden_channels,
                        ),
                    ]
                )
            )

        self.final_norm = get_normalization("rmsnorm", hidden_channels, hidden_channels)
        self.final_proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, mask, mu, t, aux, cond, cond_mask):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            text (_type_): shape (batch_size, in_channels, time)
            cond (_type_): shape (batch_size, in_channels, 32)
            cond_mask (_type_): shape (batch_size, 1, 32)

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        batch = x.shape[0]

        if t.ndim == 0:
            t = repeat(t, " -> b", b=batch)

        # time embeddings
        t = self.time_mlp(t)

        # prenet
        x = self.prenet(torch.cat([x, mu], dim=1), mask, t)

        # register tokens
        registers = repeat(self.registers, "d n -> b d n", b=batch)
        registers = cond + registers
        mask_exp = F.pad(mask, (self.num_registers, 0), value=1.0)

        # attn masks
        attn_mask = mask_exp.unsqueeze(2) * mask_exp.unsqueeze(-1)
        text_attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)

        # skip connections
        skips = []

        for idx, (block_mu, cond_block, block) in enumerate(self.blocks):
            mu = block_mu(mu, mask, text_attn_mask, t)
            x, mu = cond_block(x, mu)

            is_last_half = idx >= self.n_layers // 2

            # skip connection
            if not is_last_half:
                skips.append(x)

            if is_last_half:
                skip = skips.pop()
                x = x + skip

            x = block(x, mask, registers, attn_mask, t)

        x = self.final_norm(x)
        output = self.final_proj(x * mask)

        return output


# reference: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/decoder.py
class ProsodyDiT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        filter_channels: int,
        n_layers: int = 1,
        n_heads: int = 4,
        dim_head=None,
        num_registers: int = 32,
        kernel_size: int = 3,
        p_dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.n_layers = n_layers

        # time embeddings

        # self.time_embeddings = SinusoidalPosEmb(hidden_channels)
        self.time_mlp = TimestepEmbedding(hidden_channels, filter_channels)

        # # fourier features

        # self.fourier_module = Base2FourierFeatures(start=6, stop=8, step=1)
        # fourier_dim = (
        #     out_channels
        #     * 2
        #     * (
        #         (self.fourier_module.stop - self.fourier_module.start)
        #         // self.fourier_module.step
        #     )
        # )

        # prenet

        self.prenet = ResnetBlock1D(
            dim_in=in_channels,
            dim_out=hidden_channels,
            kernel_size=kernel_size,
            dropout=p_dropout,
            time_emb_dim=hidden_channels,
            norm_type="rmsnorm",
        )

        # cond projection
        self.cond_proj = (
            nn.Conv1d(256, hidden_channels, 1)
            if 256 != hidden_channels
            else nn.Identity()
        )

        # registers

        self.num_registers = num_registers
        self.registers = nn.Parameter(torch.zeros(hidden_channels, num_registers))
        nn.init.normal_(self.registers, std=0.02)

        # blocks

        self.blocks = nn.ModuleList()
        for idx in range(n_layers):
            self.blocks.append(
                Encoder(
                    hidden_channels=hidden_channels,
                    filter_channels=filter_channels,
                    n_heads=n_heads,
                    dim_head=dim_head,
                    kernel_size=kernel_size,
                    p_dropout=p_dropout,
                    time_emb_dim=hidden_channels,
                )
            )

        self.final_norm = get_normalization("rmsnorm", hidden_channels, hidden_channels)
        self.final_proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, mask, mu, t, aux, cond, cond_mask):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            text (_type_): shape (batch_size, in_channels, time)
            cond (_type_): shape (batch_size, in_channels, 32)
            cond_mask (_type_): shape (batch_size, 1, 32)

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        batch = x.shape[0]

        if t.ndim == 0:
            t = repeat(t, " -> b", b=batch)

        # time embeddings
        t = self.time_mlp(t)

        # prenet
        x = self.prenet(torch.cat([x, mu], dim=1), mask, t)

        # register tokens
        registers = repeat(self.registers, "d n -> b d n", b=batch)
        registers = registers + self.cond_proj(cond)
        mask_exp = F.pad(mask, (self.num_registers, 0), value=1.0)

        # attn masks
        attn_mask = mask_exp.unsqueeze(2) * mask_exp.unsqueeze(-1)

        for idx, block in enumerate(self.blocks):
            x = block(x, mask, registers, attn_mask, t)

        x = self.final_norm(x)
        output = self.final_proj(x * mask)

        return output
