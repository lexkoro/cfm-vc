import torch
import torch.nn as nn
from torch.distributions import LogisticNormal

from modules.cfm.dit import DiT


class LogitNormalTrainingTimesteps:
    def __init__(self, T=1000.0, loc=0.0, scale=1.0):
        assert T > 0
        self.T = T
        self.dist = LogisticNormal(loc, scale)

    def sample(self, size, device):
        samples = self.dist.sample(size)
        if samples is None:
            raise RuntimeError("LogisticNormal timestep sampler returned None")
        t = samples[..., 0].to(device)
        return t


class ConditionalFlowMatching(nn.Module):
    def __init__(
        self,
        in_channels=80,
        hidden_channels=192,
        filter_channels=768,
        out_channels=80,
        n_layers=6,
        n_heads=4,
        dim_head=None,
        kernel_size=3,
        p_dropout=0.05,
        use_skip_connections=False,
        sigma_min: float = 1e-06,
    ):
        super().__init__()
        self.sigma_min = sigma_min

        self.time_sampler = LogitNormalTrainingTimesteps()

        self.out_channels = out_channels

        self.estimator = DiT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            filter_channels=filter_channels,
            n_layers=n_layers,
            n_heads=n_heads,
            dim_head=dim_head,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            use_skip_connections=use_skip_connections,
        )

    def func_dphi_dt(self, x, mask, mu, cond_mel, t, guidance_scale=0.0):
        dphi_dt = self.estimator(x, mask, mu, cond_mel, t)
        if guidance_scale > 0:
            cfg_dphi_dt = self.estimator(x, mask, torch.zeros_like(mu), cond_mel, t)
            dphi_dt = (1.0 + guidance_scale) * dphi_dt - guidance_scale * cfg_dphi_dt
        return dphi_dt

    def solve_fixed_step(
        self,
        x,
        mask,
        mu,
        cond_mel,
        prompt_mask,
        t_span,
        guidance_scale,
        solver,
    ):
        t = t_span[0]
        for step in range(1, len(t_span)):
            dt = t_span[step] - t
            dphi_dt = self.func_dphi_dt(x, mask, mu, cond_mel, t, guidance_scale)

            if solver == "euler":
                x = x + dt * dphi_dt
            elif solver == "heun":
                dphi_dt_2 = self.func_dphi_dt(
                    x + dt * dphi_dt,
                    mask,
                    mu,
                    cond_mel,
                    t + dt,
                    guidance_scale,
                )
                x = x + dt * 0.5 * (dphi_dt + dphi_dt_2)
            elif solver == "midpoint":
                dphi_dt_2 = self.func_dphi_dt(
                    x + dt * 0.5 * dphi_dt,
                    mask,
                    mu,
                    cond_mel,
                    t + dt * 0.5,
                    guidance_scale,
                )
                x = x + dt * dphi_dt_2
            else:
                raise ValueError(f"Unsupported fixed-step solver: {solver}")

            # Keep prompt frames fixed so target conditioning cannot drift.
            x = torch.where(prompt_mask, cond_mel, x)

            t = t + dt

        return x

    def build_prefix_mask(self, lengths, max_len):
        positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return (positions < lengths.unsqueeze(1)).unsqueeze(1)

    @torch.no_grad()
    def inference(
        self,
        mu,
        mask,
        target_condition,
        source_lengths,
        target_lengths,
        n_timesteps,
        temperature=1.0,
        guidance_scale=0.0,
        solver="euler",
    ):
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        source_noise = (
            torch.randn(
                size=(mu.size(0), self.out_channels, source_lengths.max()),
                dtype=mu.dtype,
                device=mu.device,
            )
            * temperature
        )

        x = torch.cat([target_condition, source_noise], dim=-1)
        cond_mel = torch.cat([target_condition, torch.zeros_like(source_noise)], dim=-1)
        prompt_mask = self.build_prefix_mask(target_lengths, x.size(-1))

        assert solver in ["euler", "heun", "midpoint"], f"Unsupported solver: {solver}"

        # Ensure a clean prompt at t=0 before solving the generated region.
        x = torch.where(prompt_mask, cond_mel, x)

        return self.solve_fixed_step(
            x=x,
            mask=mask,
            mu=mu,
            cond_mel=cond_mel,
            prompt_mask=prompt_mask,
            t_span=t_span,
            guidance_scale=guidance_scale,
            solver=solver,
        )

    def sample_x0(self, x1, mask):
        x0 = torch.randn_like(x1)  # N(0,1)
        x0 = x0 * mask

        return x0

    def forward(self, x1, mask, mu, prompt_mask):
        b, _, t = mu.shape

        t = torch.rand([b, 1, 1], dtype=x1.dtype, device=x1.device, requires_grad=False)
        t = 1 - torch.cos(t * 0.5 * torch.pi)

        cfg_mask = torch.rand(b, device=x1.device) > 0.2
        mu = mu * cfg_mask.view(-1, 1, 1)

        prompt_mask = prompt_mask.bool() & mask.bool()
        generation_mask = mask.bool() & ~prompt_mask
        cond_mel = x1 * prompt_mask.to(x1.dtype)

        z = self.sample_x0(x1, mask)

        x_t = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        vector_field_estimation = self.estimator(
            x_t,
            mask,
            mu,
            cond_mel,
            t.squeeze(),
        )

        if not generation_mask.any():
            generation_mask = mask.bool()

        mse_loss = torch.nn.functional.mse_loss(
            torch.masked_select(vector_field_estimation, generation_mask),
            torch.masked_select(u, generation_mask),
        )

        return mse_loss, vector_field_estimation, generation_mask
