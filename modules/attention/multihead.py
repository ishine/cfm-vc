import math

import torch
from einops import rearrange
from torch import nn
from torch.cuda.amp import autocast


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return self.inv_freq.device

    @autocast(enabled=False)
    def forward(self, t):
        if not torch.is_tensor(t):
            t = torch.arange(t, device=self.device)

        t = t.type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@autocast(enabled=False)
def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        dim_head=None,
        p_dropout=0.0,
        cross_attn_include_queries=False,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        if dim_head is not None:
            self.dim_head = dim_head
            self.channels = dim_head * n_heads
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.cross_attn_include_queries = cross_attn_include_queries

        self.k_channels = self.channels // n_heads

        self.conv_q = torch.nn.Conv1d(channels, self.channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, self.channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, self.channels, 1)

        self.query_rotary_pe = RotaryEmbedding(self.k_channels)
        self.key_rotary_pe = RotaryEmbedding(self.k_channels)

        self.conv_o = torch.nn.Conv1d(self.channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        # length of the sequence
        t_t, t_s = x.size(2), c.size(2)

        # rotary positional embedding
        query_emb = self.query_rotary_pe(t_t)
        key_emb = self.key_rotary_pe(t_s)

        if self.cross_attn_include_queries:
            key_emb = torch.cat((query_emb, key_emb), dim=0)
            c = torch.cat((x, c), dim=-1)

        # conv
        query = self.conv_q(x)
        key = self.conv_k(c)
        value = self.conv_v(c)

        # split heads
        b, d = key.size(0), key.size(1)
        query = rearrange(query, "b (h c) t-> b h t c", h=self.n_heads)
        key = rearrange(key, "b (h c) t-> b h t c", h=self.n_heads)
        value = rearrange(value, "b (h c) t-> b h t c", h=self.n_heads)

        # apply rotary positional embedding
        query = apply_rotary_pos_emb(query_emb, query)
        key = apply_rotary_pos_emb(key_emb, key)

        # attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)

        # mask
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)

        x = self.conv_o(output)
        return x


class ConditionalMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        dim_head=None,
        p_dropout=0.0,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        if dim_head is not None:
            self.dim_head = dim_head
            self.channels = dim_head * n_heads
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout

        self.k_channels = self.channels // n_heads

        self.conv_q = torch.nn.Conv1d(channels, self.channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, self.channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, self.channels, 1)

        self.query_rotary_pe = RotaryEmbedding(self.k_channels)
        self.key_rotary_pe = RotaryEmbedding(self.k_channels)

        self.conv_o = torch.nn.Conv1d(self.channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        # length of the sequence
        t_t, t_s = x.size(2), c.size(2)

        # rotary positional embedding
        query_emb = self.query_rotary_pe(t_t)
        key_emb = self.key_rotary_pe(t_s)
        query_emb = key_emb = torch.cat((key_emb, query_emb), dim=0)
        x = c = torch.cat((c, x), dim=-1)
        t_t = x.size(2)

        # conv
        query = self.conv_q(x)
        key = self.conv_k(c)
        value = self.conv_v(c)

        # split heads
        b, d = key.size(0), key.size(1)
        query = rearrange(query, "b (h c) t-> b h t c", h=self.n_heads)
        key = rearrange(key, "b (h c) t-> b h t c", h=self.n_heads)
        value = rearrange(value, "b (h c) t-> b h t c", h=self.n_heads)

        # apply rotary positional embedding
        query = apply_rotary_pos_emb(query_emb, query)
        key = apply_rotary_pos_emb(key_emb, key)

        # attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)

        x = self.conv_o(output)

        # remove the prepended keys
        x = x[:, :, t_s:]

        return x
