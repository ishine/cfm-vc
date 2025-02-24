# Adapted from https://github.com/lucidrains/naturalspeech2-pytorch/blob/659bec7f7543e7747e809e950cc2f84242fbeec7/naturalspeech2_pytorch/naturalspeech2_pytorch.py#L532

from collections import namedtuple
from functools import wraps

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from packaging import version
from torch import einsum, nn

from modules.attention.multihead import MultiHeadAttention
from modules.modules import Conv1dGLU


def exists(val):
    return val is not None


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)

# main class


class Attend(nn.Module):
    def __init__(self, dropout=0.0, causal=False, use_flash=False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.use_flash = use_flash
        assert not (
            use_flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"

        # determine efficient attention configs for cuda and cpu
        self.config = namedtuple(
            "EfficientAttentionConfig",
            ["enable_flash", "enable_math", "enable_mem_efficient"],
        )
        self.cpu_config = self.config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once(
                "A100 GPU detected, using flash attention if input tensor is on cuda"
            )
            self.cuda_config = self.config(True, False, False)
        else:
            print_once(
                "Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda"
            )
            self.cuda_config = self.config(False, True, True)

    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def flash_attn(self, q, k, v, mask=None):
        _, heads, q_len, _, _, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = rearrange(k, "b ... -> b 1 ...").expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, "b ... -> b 1 ...").expand_as(q)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            mask = mask.expand(-1, heads, q_len, -1)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.causal,
            )

        return out

    def forward(self, q, k, v, mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device = q.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if self.use_flash:
            return self.flash_attn(q, k, v, mask=mask)

        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

        # similarity

        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # key padding mask

        if exists(mask):
            mask = rearrange(mask, "b 1 j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask

        if self.causal:
            causal_mask = self.get_mask(n, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class RMSNorm(nn.Module):
    def __init__(self, dim, scale=True):
        super().__init__()

        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x):
        gamma = default(self.gamma, 1)
        out = F.normalize(x, dim=-1) * self.scale * gamma

        return out


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4):
    dim_inner = dim * mult
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Linear(dim_inner, dim),
    )


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        n_layers=2,
        num_latents=32,
        n_heads=8,
        dim_head=64,
        ff_mult=4,
        p_dropout=0.1,
    ):
        super().__init__()

        # spectral
        self.spectral = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            nn.Mish(),
            nn.Dropout(p_dropout),
            nn.Conv1d(hidden_channels, hidden_channels, 1),
            nn.Mish(),
            nn.Dropout(p_dropout),
        )

        # temporal
        self.temporal = nn.Sequential(
            Conv1dGLU(hidden_channels, hidden_channels, 5, p_dropout),
            Conv1dGLU(hidden_channels, hidden_channels, 5, p_dropout),
        )

        # self attention
        self.attn = MultiHeadAttention(
            hidden_channels,
            hidden_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            p_dropout=p_dropout,
        )

        # latent encoder
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_channels))
        nn.init.normal_(self.latents, std=0.02)

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=hidden_channels,
                            dim_head=dim_head,
                            heads=n_heads,
                            use_flash=False,
                            cross_attn_include_queries=True,
                        ),
                        FeedForward(dim=hidden_channels, mult=ff_mult),
                    ]
                )
            )

        self.norm = RMSNorm(hidden_channels)

    def forward(self, x, x_mask):
        batch = x.shape[0]

        # spectral & temporal
        x = self.spectral(x) * x_mask
        x = self.temporal(x) * x_mask

        # self attn
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = self.attn(x, c=x, attn_mask=attn_mask) + x

        x = rearrange(x, "b d n -> b n d")

        # latents
        latents = repeat(self.latents, "n d -> b n d", b=batch)
        latents_mask = torch.ones(
            batch, 1, latents.shape[1], device=x.device, dtype=x.dtype
        )

        # latent encoder
        conds_mask = torch.cat([latents_mask, x_mask], dim=-1).bool()
        for attn, ff in self.layers:
            latents = attn(latents, x, mask=conds_mask) + latents
            latents = ff(latents) + latents

        # norm
        latents = self.norm(latents)
        latents = rearrange(latents, "b n d -> b d n")

        return latents, latents_mask


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_context=None,
        causal=False,
        dim_head=64,
        heads=8,
        dropout=0.0,
        use_flash=False,
        cross_attn_include_queries=False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)

        # norm
        self.norm_q = RMSNorm(dim)
        self.norm_kv = RMSNorm(dim_context)

        self.attend = Attend(causal=causal, dropout=dropout, use_flash=use_flash)
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x, context=None, mask=None):
        h, has_context = self.heads, exists(context)

        context = default(context, x)

        # norm
        x = self.norm_q(x)
        context = self.norm_kv(context)

        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim=-2)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        out = self.attend(q, k, v, mask=mask)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
