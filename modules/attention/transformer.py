from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from modules.attention.ffn import FFN
from modules.attention.multihead import (
    ConditionalMultiHeadAttention,
    MultiHeadAttention,
)
from modules.modules import NORM_TYPES, FiLM, LayerNorm, get_normalization


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class GEGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x, gate = x.chunk(2, dim=self.dim)
        return F.gelu(gate) * x


class ConditionalCrossAttnBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        n_heads: int,
        dim_head: Optional[int] = None,
        p_dropout: float = 0.0,
        utt_emb_dim: int = 0,
    ):
        super().__init__()

        # modulation
        self.modulation = nn.Sequential(
            nn.Linear(utt_emb_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 6 * hidden_channels),
        )

        # conditional self attn
        self.norm_1 = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.attn = ConditionalMultiHeadAttention(
            channels=hidden_channels,
            out_channels=hidden_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            p_dropout=p_dropout,
        )
        # film layer
        self.norm_2 = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.film = FiLM(hidden_channels, hidden_channels)

    def forward(self, x, x_mask, cond, cond_mask, utt_emb):
        # attn mask
        cond_mask = torch.cat([cond_mask, x_mask], dim=-1)
        attn_mask = cond_mask.unsqueeze(2) * cond_mask.unsqueeze(-1)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(utt_emb).unsqueeze(-1).chunk(6, dim=1)
        )  # shape: [batch_size, channel, 1]

        attn_in = self.modulate(
            self.norm_1(x.transpose(1, 2)).transpose(1, 2), shift_msa, scale_msa
        )
        x = gate_msa * self.attn(attn_in, c=cond, attn_mask=attn_mask) + x
        x = x * x_mask

        film_in = self.modulate(
            self.norm_2(x.transpose(1, 2)).transpose(1, 2), shift_mlp, scale_mlp
        )
        x = gate_mlp * self.film(film_in, x) + x
        x = x * x_mask

        return x

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift


class ConditionalTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        dim_head: Optional[int] = None,
        p_dropout: float = 0.0,
        cond_emb_dim: int = 0,
        causal_ffn: bool = False,
        norm_type: NORM_TYPES = "rmsnorm",
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_heads = n_heads
        self.dim_hidden = hidden_channels * 2

        # self attn
        self.norm_1 = get_normalization(norm_type, hidden_channels, cond_emb_dim)
        self.cond_norm = get_normalization(norm_type, hidden_channels, cond_emb_dim)
        self.attn = ConditionalMultiHeadAttention(
            channels=hidden_channels,
            out_channels=hidden_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            p_dropout=p_dropout,
        )

        # feed forward
        self.norm_2 = get_normalization(norm_type, hidden_channels, cond_emb_dim)
        self.ff = FFN(
            hidden_channels,
            hidden_channels,
            filter_channels,
            p_dropout=p_dropout,
            causal=causal_ffn,
            use_glu=True,
        )

    def forward(self, x, x_mask, cond, cond_attn_mask, cond_emb=None):
        # self attn
        attn_in = self.norm_1(x, cond_emb)
        cond = self.cond_norm(cond, cond_emb)
        x = self.attn(attn_in, c=cond, attn_mask=cond_attn_mask) + x

        # feed forward
        ff_in = self.norm_2(x, cond_emb)
        x = self.ff(ff_in, x_mask) + x

        return x * x_mask


class ConditionalEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        dim_head: Optional[int] = None,
        n_layers: int = 6,
        p_dropout: float = 0.0,
        norm_type: NORM_TYPES = "rmsnorm",
        cond_emb_dim: int = 0,
        causal_ffn: bool = False,
        use_final_norm: bool = True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            # add conditional cross attn block
            self.layers.append(
                ConditionalTransformerBlock(
                    hidden_channels=hidden_channels,
                    filter_channels=filter_channels,
                    n_heads=n_heads,
                    dim_head=dim_head,
                    p_dropout=p_dropout,
                    cond_emb_dim=cond_emb_dim,
                    causal_ffn=causal_ffn,
                    norm_type=norm_type,
                )
            )

        self.final_norm = (
            get_normalization("rmsnorm", hidden_channels)
            if use_final_norm
            else nn.Identity()
        )

    def forward(
        self,
        x,
        x_mask,
        cond,
        cond_mask,
        cond_emb=None,
    ):
        # attn masks
        cond_mask = torch.cat([cond_mask, x_mask], dim=-1)
        cond_attn_mask = cond_mask.unsqueeze(2) * cond_mask.unsqueeze(-1)

        x = x * x_mask

        for idx, layer in enumerate(self.layers):
            # use prosody condition
            x = layer(x, x_mask, cond, cond_attn_mask, cond_emb)

        x = self.final_norm(x * x_mask)

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        dim_head: Optional[int] = None,
        p_dropout: float = 0.0,
        cond_emb_dim: int = 0,
        norm_type: NORM_TYPES = "rmsnorm",
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout

        # self attention
        self.self_norm = get_normalization(norm_type, hidden_channels, cond_emb_dim)
        self.attn = MultiHeadAttention(
            hidden_channels,
            hidden_channels,
            n_heads,
            dim_head=dim_head,
            p_dropout=p_dropout,
        )

        # ff layer
        self.ff_norm = get_normalization(norm_type, hidden_channels, cond_emb_dim)
        self.ffn = FFN(
            hidden_channels,
            hidden_channels,
            filter_channels,
            p_dropout=p_dropout,
            use_glu=True,
        )

    def forward(self, x, x_mask, attn_mask, cond_emb=None):
        x = x * x_mask

        attn_in = self.self_norm(x, cond_emb)
        x = self.attn(attn_in, c=attn_in, attn_mask=attn_mask) + x

        # feed-forward
        ffn_in = self.ff_norm(x, cond_emb)
        x = self.ffn(ffn_in, x_mask) + x

        return x * x_mask


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        dim_head: Optional[int] = None,
        kernel_size: int = 3,
        p_dropout: float = 0.0,
        utt_emb_dim: int = 0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels

        # modulation
        self.modulation = nn.Sequential(
            nn.Linear(utt_emb_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 6 * hidden_channels),
        )

        # conditional self attn
        self.norm_1 = nn.LayerNorm(hidden_channels, eps=1e-6, elementwise_affine=False)
        self.attn = MultiHeadAttention(
            channels=hidden_channels,
            out_channels=hidden_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            p_dropout=p_dropout,
        )

        # feed forward
        self.norm_2 = nn.LayerNorm(hidden_channels, eps=1e-6, elementwise_affine=False)
        self.ff = FFN(
            hidden_channels,
            hidden_channels,
            filter_channels,
            p_dropout=p_dropout,
        )

    def forward(self, x, x_mask, attn_mask, utt_emb):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(utt_emb).unsqueeze(-1).chunk(6, dim=1)
        )  # shape: [batch_size, channel, 1]

        # conditional self attn
        attn_in = self.modulate(
            self.norm_1(x.transpose(1, 2)).transpose(1, 2), shift_msa, scale_msa
        )
        x = gate_msa * self.attn(attn_in, c=attn_in, attn_mask=attn_mask) + x

        # feed forward
        ff_in = self.modulate(
            self.norm_2(x.transpose(1, 2)).transpose(1, 2), shift_mlp, scale_mlp
        )
        x = gate_mlp * self.ff(ff_in, x_mask) + x

        return x * x_mask

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift


class DiTEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        dim_head: Optional[int] = None,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        utt_emb_dim: int = 0,
        use_cond_block_at: list[int] = [],
        use_final_norm: bool = True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.blocks = nn.ModuleList()
        for i in range(self.n_layers):
            if i in use_cond_block_at:
                # add conditional cross attn block
                self.blocks.append(
                    DiTBlock(
                        hidden_channels=hidden_channels,
                        filter_channels=filter_channels,
                        n_heads=n_heads,
                        dim_head=dim_head,
                        kernel_size=kernel_size,
                        p_dropout=p_dropout,
                        utt_emb_dim=utt_emb_dim,
                    )
                )
            else:
                # self attention
                self.blocks.append(
                    TransformerBlock(
                        hidden_channels=hidden_channels,
                        filter_channels=filter_channels,
                        n_heads=n_heads,
                        dim_head=dim_head,
                        kernel_size=kernel_size,
                        p_dropout=p_dropout,
                    )
                )

        self.final_norm = (
            LayerNorm(hidden_channels) if use_final_norm else nn.Identity()
        )

    def forward(self, x, x_mask, utt_emb=None):
        # attn mask
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask

        # encoder
        for layer in self.blocks:
            if isinstance(layer, DiTBlock):
                x = layer(x, x_mask, attn_mask, utt_emb)
            else:
                x = layer(x, x_mask, attn_mask)

        x = self.final_norm(x * x_mask)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        dim_head: Optional[int] = None,
        kernel_size: int = 3,
        p_dropout: float = 0.0,
        utt_emb_dim: int = 0,
        causal_ffn: bool = False,
        norm_type: str = "rmsnorm",
        use_final_norm: bool = True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.attn_layers = nn.ModuleList()

        self.norm_layers_1 = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        self.ffn_layers_1 = nn.ModuleList()

        for i in range(self.n_layers):
            # self attention
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    dim_head=dim_head,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_1.append(
                get_normalization(norm_type, hidden_channels, utt_emb_dim)
            )

            # ff layer
            self.ffn_layers_1.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size=kernel_size,
                    p_dropout=p_dropout,
                    causal=causal_ffn,
                    use_glu=True,
                )
            )
            self.norm_layers_2.append(
                get_normalization(norm_type, hidden_channels, utt_emb_dim)
            )

        self.final_norm = (
            LayerNorm(hidden_channels) if use_final_norm else nn.Identity()
        )

    def forward(self, x, x_mask, g=None):
        # attn mask
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask

        for i in range(self.n_layers):
            # self-attention
            self_attn_in = self.norm_layers_1[i](x, g)
            x = self.attn_layers[i](self_attn_in, self_attn_in, attn_mask) + x

            # feed-forward
            ffn_in = self.norm_layers_2[i](x, g)
            x = self.ffn_layers_1[i](ffn_in, x_mask) + x

        x = self.final_norm(x * x_mask)

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        dim_head: Optional[int] = None,
        kernel_size: int = 3,
        p_dropout: float = 0.0,
        utt_emb_dim: int = 0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels

        # modulation
        self.modulation = nn.Sequential(
            nn.Linear(utt_emb_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 6 * hidden_channels),
        )

        # self attn
        self.norm_1 = nn.LayerNorm(hidden_channels, eps=1e-6, elementwise_affine=False)
        self.attn = MultiHeadAttention(
            channels=hidden_channels,
            out_channels=hidden_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            p_dropout=p_dropout,
        )

        # cross attn
        self.norm_2 = LayerNorm(hidden_channels)
        self.cross_attn = MultiHeadAttention(
            channels=hidden_channels,
            out_channels=hidden_channels,
            n_heads=n_heads,
            dim_head=dim_head,
            p_dropout=p_dropout,
        )

        # feed forward
        self.norm_3 = nn.LayerNorm(hidden_channels, eps=1e-6, elementwise_affine=False)
        self.ff = FFN(
            hidden_channels,
            hidden_channels,
            filter_channels,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
        )

    def forward(self, x, x_mask, h, attn_mask, cond_attn_mask, utt_emb):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(utt_emb).unsqueeze(-1).chunk(6, dim=1)
        )  # shape: [batch_size, channel, 1]

        # self attn
        attn_in = self.modulate(
            self.norm_1(x.transpose(1, 2)).transpose(1, 2), shift_msa, scale_msa
        )
        x = gate_msa * self.attn(attn_in, c=attn_in, attn_mask=attn_mask) + x

        # cross attn
        cross_in = self.norm_2(x)
        x = self.cross_attn(cross_in, c=h, attn_mask=cond_attn_mask) + x

        # feed forward
        ff_in = self.modulate(
            self.norm_3(x.transpose(1, 2)).transpose(1, 2), shift_mlp, scale_mlp
        )
        x = gate_mlp * self.ff(ff_in, x_mask) + x

        return x * x_mask

    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale) + shift


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        dim_head: Optional[int] = None,
        kernel_size: int = 3,
        p_dropout: float = 0.0,
        utt_emb_dim: int = 0,
        use_final_norm: bool = True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.blocks = nn.ModuleList()
        for i in range(self.n_layers):
            self.blocks.append(
                DecoderBlock(
                    hidden_channels=hidden_channels,
                    filter_channels=filter_channels,
                    n_heads=n_heads,
                    dim_head=dim_head,
                    kernel_size=kernel_size,
                    p_dropout=p_dropout,
                    utt_emb_dim=utt_emb_dim,
                )
            )

        self.final_norm = (
            LayerNorm(hidden_channels) if use_final_norm else nn.Identity()
        )

    def forward(self, x, x_mask, h, h_mask, utt_emb=None):
        """
        x: decoder input
        h: encoder output
        """

        # attn mask
        attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask

        # encoder
        for layer in self.blocks:
            x = layer(x, x_mask, h, attn_mask, utt_emb)

        x = self.final_norm(x * x_mask)

        return x
