import torch
import torch.nn as nn
from torchdyn.core import NeuralODE

from modules.cfm.dit import DiT


class Wrapper(nn.Module):
    def __init__(
        self, vector_field_net, mask, mu, spk, cond, cond_mask, guidance_scale
    ):
        super(Wrapper, self).__init__()
        self.net = vector_field_net
        self.mask = mask
        self.mu = mu
        self.spk = spk
        self.cond = cond
        self.cond_mask = cond_mask
        self.guidance_scale = guidance_scale

    def forward(self, t, x, args):
        # NOTE: args cannot be dropped here. This function signature is strictly required by the NeuralODE class
        t = torch.tensor([t], device=t.device)

        dphi_dt = self.net(
            x, self.mask, self.mu, t, self.spk, self.cond, self.cond_mask
        )

        # if self.guidance_scale > 0:
        #     mu_avg = self.mu.mean(2, keepdims=True).expand_as(self.mu)
        #     dphi_avg = self.net(
        #         x, self.mask, mu_avg, t, self.spk, self.cond, self.cond_mask
        #     )
        #     dphi_dt = dphi_dt + self.guidance_scale * (dphi_dt - dphi_avg)

        # Classifier-Free Guidance inference introduced in VoiceBox
        if self.guidance_scale > 0:
            cfg_dphi_dt = self.net(
                x,
                self.mask,
                torch.zeros_like(self.mu),
                t,
                self.spk,
                torch.zeros_like(self.cond),
                self.cond_mask,
            )
            dphi_dt = (
                1.0 + self.guidance_scale
            ) * dphi_dt - self.guidance_scale * cfg_dphi_dt

        return dphi_dt


class ConditionalFlowMatching(nn.Module):
    def __init__(
        self,
        estimator_params={
            "in_channels": 80,
            "hidden_channels": 192,
            "out_channels": 80,
            "filter_channels": 768,
            "dropout": 0.05,
            "n_layers": 6,
            "n_heads": 4,
            "dim_head": None,
            "kernel_size": 3,
        },
        sigma_min: float = 1e-06,
    ):
        super(ConditionalFlowMatching, self).__init__()
        self.sigma_min = sigma_min

        self.out_channels = estimator_params["out_channels"]

        self.estimator = DiT(
            in_channels=estimator_params["in_channels"],
            hidden_channels=estimator_params["hidden_channels"],
            out_channels=estimator_params["out_channels"],
            filter_channels=estimator_params["filter_channels"],
            n_layers=estimator_params["n_layers"],
            n_heads=estimator_params["n_heads"],
            dim_head=estimator_params["dim_head"],
            num_registers=32,
            kernel_size=estimator_params["kernel_size"],
            p_dropout=estimator_params["dropout"],
        )

    def ode_wrapper(self, mask, mu, spk, cond, cond_mask, guidance_scale):
        # self.estimator receives x, mask, mu, t, spk as arguments
        return Wrapper(self.estimator, mask, mu, spk, cond, cond_mask, guidance_scale)

    @torch.no_grad()
    def inference(
        self,
        z,
        mask,
        mu,
        n_timesteps,
        spk=None,
        cond=None,
        cond_mask=None,
        guidance_scale=0.0,
        solver="dopri5",
    ):
        t_span = torch.linspace(
            0, 1, n_timesteps + 1, device=mu.device
        )  # NOTE: n_timesteps means n+1 points in [0, 1]
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        neural_ode = NeuralODE(
            self.ode_wrapper(mask, mu, spk, cond, cond_mask, guidance_scale),
            solver=solver,
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
        )

        x = z
        _, traj = neural_ode(x, t_span)

        return traj[-1]

    def sample_x0(self, mu, mask):
        x0 = torch.randn_like(mu)  # N(0,1)
        x0 = x0 * mask

        return x0

    def forward(self, x1, mask, mu, spk, cond, cond_mask):
        b, _, t = mu.shape

        t = torch.rand([b, 1, 1], dtype=x1.dtype, device=x1.device, requires_grad=False)
        t = 1 - torch.cos(t * 0.5 * torch.pi)

        cfg_mask = torch.rand(b, device=x1.device) > 0.2
        mu = mu * cfg_mask.view(-1, 1, 1)
        cond = cond * cfg_mask.view(-1, 1, 1)

        return self.loss_t(x1, mask, mu, t, spk, cond, cond_mask)

    def loss_t(self, x1, mask, mu, t, spk, cond, cond_mask):
        # sample noise p(x_0)
        z = self.sample_x0(x1, mask)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        vector_field_estimation = self.estimator(
            y, mask, mu, t.squeeze(), spk, cond, cond_mask
        )

        mse_loss = torch.nn.functional.mse_loss(
            torch.masked_select(vector_field_estimation, mask.bool()),
            torch.masked_select(u, mask.bool()),
        )

        return mse_loss, vector_field_estimation


# class OTCFM(ConditionalFlowMatching):
#     def __init__(
#         self,
#         estimator_params={
#             "in_channels": 80,
#             "hidden_channels": 192,
#             "out_channels": 80,
#             "filter_channels": 768,
#             "dropout": 0.05,
#             "n_layers": 6,
#             "n_heads": 4,
#             "dim_head": None,
#             "kernel_size": 3,
#         },
#         sigma_min: float = 1e-06,
#     ):
#         super(OTCFM, self).__init__(
#             estimator_params=estimator_params, sigma_min=sigma_min
#         )

#         # self.ot_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=0)
#         self.ot_sampler = OTPlanSampler(method="exact")

#     def loss_t(self, x1, mask, mu, t, spk, cond, cond_mask):
#         x0 = self.sample_x0(mu, mask)

#         # x1 and x0 shape is [B, 80, L]
#         B, D, L = x0.shape
#         new_x0 = torch.zeros_like(x1)
#         new_x1 = torch.zeros_like(x0)
#         for l in range(L):
#             sub_x0, sub_x1, i, j = self.ot_sampler.sample_plan_with_index(
#                 x1[..., l], x0[..., l], replace=False
#             )
#             index_that_would_sort_i = np.argsort(
#                 i
#             )  # To keep i and j synchronized for each position in L
#             i = i[index_that_would_sort_i]
#             j = j[index_that_would_sort_i]

#             new_x0[..., l] = x1[i, :, l]
#             new_x1[..., l] = x0[j, :, l]

#         x1 = new_x0
#         x0 = new_x1

#         ut = x1 - x0  # conditional vector field. This is actually x0 - x1 in paper.
#         mu_t = t * x1 + (1 - t) * x0  # conditional Gaussian mean
#         sigma_t = self.sigma_min
#         x = mu_t + sigma_t * torch.randn_like(x1)  # sample p_t(x|x_0, x_1)

#         vector_field_estimation = self.estimator(
#             x, mask, mu, t.squeeze(), spk, cond, cond_mask
#         )

#         mse_loss = torch.nn.functional.mse_loss(
#             torch.masked_select(vector_field_estimation, mask.bool()),
#             torch.masked_select(ut, mask.bool()),
#         )

#         return mse_loss, vector_field_estimation
