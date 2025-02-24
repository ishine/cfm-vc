import torch
from torch import nn

from modules.attention.transformer import ConditionalEncoder
from modules.cfm.cfm_euler import CFMDecoder
from modules.modules import AdainResBlk1d, ResnetBlock1D


class ConditionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, d_model, style_dim=512):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=d_model)
        self.adain = AdainResBlk1d(
            dim_in=d_model,
            dim_out=d_model,
            style_dim=style_dim,
            kernel_size=3,
            actv=nn.SiLU(),
        )

    def forward(self, x, x_mask, utt_emb):
        emb = self.embed(x).transpose(1, 2)
        x = self.adain(emb, utt_emb)
        return x * x_mask


class AuxDecoder(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        kernel_size,
        n_layers,
        n_heads,
        dim_head=None,
        p_dropout=0.0,
    ):
        super().__init__()

        self.aux_prenet = nn.Conv1d(
            input_channels, hidden_channels, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.prenet = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )

        # encoder
        self.aux_decoder = ConditionalEncoder(
            hidden_channels=hidden_channels,
            filter_channels=hidden_channels * 4,
            n_heads=n_heads,
            dim_head=dim_head,
            n_layers=n_layers,
            p_dropout=p_dropout,
            causal_ffn=True,
            norm_type="rmsnorm",
            use_final_norm=True,
        )

        self.proj = nn.Conv1d(hidden_channels, output_channels, 1)

    def forward(self, x, x_mask, aux, cond, cond_mask):
        # detach x
        x = torch.detach(x)

        # prenets
        x = x + self.aux_prenet(aux) * x_mask
        x = self.prenet(x) * x_mask

        # attention
        x = self.aux_decoder(x, x_mask, cond, cond_mask)

        # out projection
        x = self.proj(x) * x_mask

        return x * x_mask


class VariancePredictorCFM(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 192,
        estimator_params={},
    ):
        """
        Initialize variance predictor module.

        """
        super().__init__()

        self.aux_prenet = nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1)
        self.prenet = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )

        # aux conv
        self.aux_embedding = ResnetBlock1D(
            dim_in=1,
            dim_out=hidden_channels,
            utt_emb_dim=hidden_channels,
            norm_type="adarmsnorm",
        )

        # decoder
        self.decoder = CFMDecoder(
            in_channels=estimator_params["in_channels"] + hidden_channels,
            hidden_channels=estimator_params["hidden_channels"],
            out_channels=estimator_params["out_channels"],
            filter_channels=estimator_params["filter_channels"],
            n_heads=estimator_params["n_heads"],
            dim_head=estimator_params["dim_head"],
            n_layers=estimator_params["n_layers"],
            kernel_size=estimator_params["kernel_size"],
            p_dropout=estimator_params["dropout"],
        )

    def forward(self, x, x_mask, cond, cond_mask, aux_target):
        x = x.detach()

        # prenets
        x = x + self.aux_prenet(aux_target) * x_mask
        x = self.prenet(x) * x_mask

        # Compute loss of score-based decoder
        diff_loss, _ = self.decoder.compute_loss(
            aux_target,
            x_mask,
            x,
            c=None,
            cond=cond,
            cond_mask=cond_mask,
        )

        cond_mean = torch.mean(cond, dim=2)
        aux_embedding = self.aux_embedding(aux_target, x_mask, cond_mean)

        return diff_loss, aux_embedding

    @torch.no_grad()
    def inference(
        self,
        x,
        x_mask,
        aux_source,
        cond,
        cond_mask,
        n_timesteps=2,
        temperature=1.0,
        solver="euler",
    ):
        # prenets
        x = x + self.aux_prenet(aux_source) * x_mask
        x = self.prenet(x) * x_mask

        # decoder
        aux_pred = self.decoder.forward(
            x,
            x_mask,
            n_timesteps,
            temperature=temperature,
            c=None,
            cond=cond,
            cond_mask=cond_mask,
            solver=solver,
        )

        cond_mean = torch.mean(cond, dim=2)
        aux_embedding = self.aux_embedding(aux_pred, x_mask, cond_mean)

        return aux_embedding
