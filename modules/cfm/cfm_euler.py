# modified from https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/flow_matching.py
import torch

from modules.cfm.dit import ProsodyDiT


class CFMDecoder(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        filter_channels,
        n_heads,
        dim_head,
        n_layers,
        kernel_size,
        p_dropout,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.sigma_min = 1e-06

        self.estimator = ProsodyDiT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            filter_channels=filter_channels,
            n_layers=n_layers,
            n_heads=n_heads,
            dim_head=dim_head,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
        )

    @torch.inference_mode()
    def forward(
        self,
        mu,
        mask,
        n_timesteps,
        temperature=1.0,
        c=None,
        cond=None,
        cond_mask=None,
        solver="euler",
    ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        size = list(mu.size())
        size[1] = self.out_channels
        z = torch.randn(size=size).to(mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        if solver == "euler":
            return self.solve_euler(
                z,
                t_span=t_span,
                mu=mu,
                mask=mask,
                c=c,
                cond=cond,
                cond_mask=cond_mask,
            )
        elif solver == "heun":
            return self.solve_heun(
                z,
                t_span=t_span,
                mu=mu,
                mask=mask,
                c=c,
                cond=cond,
                cond_mask=cond_mask,
            )
        elif solver == "midpoint":
            return self.solve_midpoint(
                z,
                t_span=t_span,
                mu=mu,
                mask=mask,
                c=c,
                cond=cond,
                cond_mask=cond_mask,
            )

    def solve_euler(self, x, t_span, mu, mask, c, cond, cond_mask):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = self.func_dphi_dt(x, mask, mu, t, c, cond, cond_mask)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def solve_heun(self, x, t_span, mu, mask, c, cond, cond_mask):
        """
        Fixed heun solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # -! : reserved space for debugger
        sol = []
        steps = 1

        while steps <= len(t_span) - 1:
            dphi_dt = self.func_dphi_dt(x, mask, mu, t, c, cond, cond_mask)
            dphi_dt_2 = self.func_dphi_dt(
                x + dt * dphi_dt, mask, mu, t + dt, c, cond, cond_mask
            )

            # - Euler's -> Y'n+1' = Y'n' + h * F(X'n', Y'n')
            # x = x + dt * dphi_dt

            # - Heun's -> Y'n+1' = Y'n' + h * 0.5( F(X'n', Y'n') + F(X'n' + h, Y'n' + h * F(X'n', Y'n') ) )
            x = x + dt * 0.5 * (dphi_dt + dphi_dt_2)
            t = t + dt

            sol.append(x)
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1

        return sol[-1]

    def solve_midpoint(self, x, t_span, mu, mask, c, cond, cond_mask):
        """
        Fixed midpoint solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # -! : reserved space for debugger
        sol = []
        steps = 1

        while steps <= len(t_span) - 1:
            dphi_dt = self.func_dphi_dt(x, mask, mu, t, c, cond, cond_mask)
            dphi_dt_2 = self.func_dphi_dt(
                x + dt * 0.5 * dphi_dt, mask, mu, t + dt * 0.5, c, cond, cond_mask
            )

            # - Euler's -> Y'n+1' = Y'n' + h * F(X'n', Y'n')
            # x = x + dt * dphi_dt

            # - midpoint -> Y'n+1' = Y'n' + h * F(X'n' + 0.5 * h, Y'n' + 0.5 * h * F(X'n', Y'n') )
            x = x + dt * dphi_dt_2
            t = t + dt

            sol.append(x)
            if steps < len(t_span) - 1:
                dt = t_span[steps + 1] - t
            steps += 1

        return sol[-1]

    def func_dphi_dt(self, x, mask, mu, t, c, cond, cond_mask):
        dphi_dt = self.estimator(x, mask, mu, t, c, cond, cond_mask)
        return dphi_dt

    def compute_loss(self, x1, mask, mu, c, cond, cond_mask):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            c (torch.Tensor, optional): speaker condition.

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        t = 1 - torch.cos(t * 0.5 * torch.pi)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        estimator_out = self.estimator(y, mask, mu, t.squeeze(), c, cond, cond_mask)
        mse_loss = torch.nn.functional.mse_loss(
            estimator_out.masked_select(mask.bool()),
            u.masked_select(mask.bool()),
        )

        return mse_loss, y
