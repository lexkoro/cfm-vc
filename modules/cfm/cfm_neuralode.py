import torch
import torch.nn as nn
from torchdyn.core import NeuralODE

from modules.cfm.decoder import Decoder
from modules.cfm.diffusion import GradLogPEstimator2d
from modules.cfm.dit import DiT
from modules.cfm.wavenet import WaveNet


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

        if self.guidance_scale > 0.0:
            mu_avg = self.mu.mean(2, keepdims=True).expand_as(self.mu)
            dphi_avg = self.net(
                x, self.mask, mu_avg, t, self.spk, self.cond, self.cond_mask
            )
            dphi_dt = dphi_dt + self.guidance_scale * (dphi_dt - dphi_avg)

        return dphi_dt


class FM(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channel,
        spk_emb_dim=64,
        estimator="diffusion",
        sigma_min: float = 0.1,
    ):
        super(FM, self).__init__()
        self.n_feats = in_channels
        self.out_channel = out_channel
        self.spk_emb_dim = spk_emb_dim
        self.sigma_min = sigma_min

        if estimator == "diffusion":
            print("Using Diffusion")
            self.estimator = GradLogPEstimator2d(
                dim=128,
                dim_mults=(1, 2, 4),
                groups=8,
                spk_emb_dim=spk_emb_dim,
                pe_scale=1000,
            )
        elif estimator == "wavenet":
            print("Using WaveNet")
            self.estimator = WaveNet(
                kernel_size=3,
                layers=18,
                stacks=3,
                cross_attn_per_layer=3,
                base_dilation=2,
                input_channels=in_channels,
                output_channels=out_channel,
                residual_channels=256,
                gate_channels=512,
                skip_channels=256,
                global_channels=spk_emb_dim,
                dropout_rate=0.0,
                bias=True,
                use_weight_norm=False,
                scale_residual=False,
                scale_skip_connect=False,
            )
        elif estimator == "dit":
            print("Using DiT")
            self.estimator = DiT(
                in_channels=in_channels * 2,
                hidden_channels=hidden_channels,
                out_channels=out_channel,
                filter_channels=hidden_channels * 4,
                dropout=0.00,
                n_layers=6,
                n_heads=2,
                kernel_size=3,
                utt_emb_dim=spk_emb_dim,
            )
        elif estimator == "decoder":
            print("Using Decoder")
            self.estimator = Decoder(
                in_channels=in_channels * 2,
                out_channels=out_channel,
                channels=(256, 256),
                dropout=0.05,
                attention_head_dim=64,
                n_blocks=1,
                num_mid_blocks=2,
                num_heads=4,
                act_fn="snakebeta",
            )
        else:
            raise NotImplementedError

        self.criterion = torch.nn.MSELoss()

    def ode_wrapper(self, mask, mu, spk, cond, cond_mask, guidance_scale=0.0):
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
        solver="dopri5",
        guidance_scale=0.0,
    ):
        t_span = torch.linspace(
            0, 1, n_timesteps + 1
        )  # NOTE: n_timesteps means n+1 points in [0, 1]
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

    def forward(self, x1, mask, mu, spk=None, cond=None, cond_mask=None, offset=1e-5):
        t = torch.rand(
            x1.shape[0], dtype=x1.dtype, device=x1.device, requires_grad=False
        )
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x1, mask, mu, t, spk, cond, cond_mask)

    def loss_t(self, x1, mask, mu, t, spk=None, cond=None, cond_mask=None):
        t_unsqueeze = t.unsqueeze(1).unsqueeze(1)
        mu_t = t_unsqueeze * x1
        sigma_t = 1 - (1 - self.sigma_min) * t_unsqueeze
        x = mu_t + sigma_t * torch.randn_like(x1)  # sample p_t(x|x_0, x_1)
        ut = (self.sigma_min - 1) / sigma_t * (x - mu_t) + x1

        vector_field_estimation = self.estimator(x, mask, mu, t, spk, cond, cond_mask)
        mse_loss = self.criterion(ut, vector_field_estimation)
        return mse_loss, x

    @property
    def nparams(self):
        return sum([p.numel() for p in self.parameters()])


class ConditionalFlowMatching(FM):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        spk_emb_dim=64,
        estimator="diffusion",
        sigma_min=0.1,
        cond_by_mu=True,
    ):
        super(ConditionalFlowMatching, self).__init__(
            in_channels,
            hidden_channels,
            out_channels,
            spk_emb_dim,
            estimator,
            sigma_min,
        )
        self.cond_by_mu = cond_by_mu

    def sample_x0(self, mu, mask):
        x0 = torch.randn_like(mu)  # N(0,1)
        mask = mask.bool()
        x0.masked_fill_(~mask, 0)
        return x0

    def forward(
        self, x1, noise, mask, mu, spk=None, cond=None, cond_mask=None, offset=1e-5
    ):
        t = torch.rand(
            x1.shape[0], dtype=x1.dtype, device=x1.device, requires_grad=False
        )
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x1, noise, mask, mu, t, spk, cond, cond_mask)

    def loss_t(self, x1, noise, mask, mu, t, spk=None, cond=None, cond_mask=None):
        # construct noise (in CFM theory, this is x0)
        if noise is not None:
            x0 = noise
        else:
            x0 = self.sample_x0(mu, mask)

        ut = x1 - x0  # conditional vector field.
        t_unsqueeze = t.unsqueeze(1).unsqueeze(1)
        mu_t = t_unsqueeze * x1 + (1 - t_unsqueeze) * x0  # conditional Gaussian mean
        x = mu_t + self.sigma_min * torch.randn_like(x1)  # sample p_t(x|x_0, x_1)

        vector_field_estimation = self.estimator(x, mask, mu, t, spk, cond, cond_mask)
        mse_loss = self.criterion(ut, vector_field_estimation)
        return mse_loss, x

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
        solver="dopri5",
        guidance_scale=0.0,
    ):
        super_class = super()
        return super_class.inference(
            z,
            mask,
            mu,
            n_timesteps,
            spk=spk,
            cond=cond,
            cond_mask=cond_mask,
            solver=solver,
            guidance_scale=guidance_scale,
        )
