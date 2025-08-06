import numpy as np
import torch
from torch import nn
import pdb
import json

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
    catmull_rom_spline_with_rotation,
    SplineLoss
)

class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', scales_path=None, clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        print(f"Alphas Cumprod: {alphas_cumprod}")
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)
        if isinstance(self.loss_fn, SplineLoss):
            assert scales_path is not None, "The scales for the normalized trajectories need to be provided for metric scale!" 
            self.loss_fn.scales = json.load(open(scales_path))

        # Get bottleneck activation
        self.get_bottleneck_feats = False
        self.bottleneck_feats = None


    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float64)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return noise            
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t[:, :, self.action_dim:]
        )

        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, visual_cond, t):
        if self.get_bottleneck_feats:
            noise, self.bottleneck_feats = self.model(x, cond, visual_cond, t)
        else:
            noise, _ = self.model(x, cond, visual_cond, t)

        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, visual_cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, visual_cond=visual_cond, t=t)
        noise = torch.randn_like(x[:, :, self.action_dim:])
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, visual_cond, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        # Extract known indices and values
        known_indices = np.array(list(cond.keys()), dtype=int)

        # candidate_no x batch_size x dim
        known_values = np.stack([c.cpu().numpy() for c in cond.values()], axis=0)
        known_values = np.moveaxis(known_values, 0, 1)

        sorted_indices = np.argsort(known_indices)
        known_indices = known_indices[sorted_indices]
        known_values = known_values[:, sorted_indices]

        # Build the structured spline guess
        catmull_spline_trajectory = np.array([
            catmull_rom_spline_with_rotation(known_values[b, :, :-1], known_indices, shape[1]) 
            for b in range(batch_size)
        ])
        catmull_spline_trajectory = torch.tensor(
            catmull_spline_trajectory, 
            dtype=torch.float64, 
            device=device
        )            


        if self.predict_epsilon:
            x = torch.randn((shape[0], shape[1], self.observation_dim), device=device, dtype=torch.float64)
            cond_residual = {k: torch.zeros_like(v)[:, :-1] for k, v in cond.items()}
            is_cond = torch.zeros((shape[0], shape[1], 1), device=device, dtype=torch.float64)
            is_cond[:, known_indices, :] = 1.0             

        if return_diffusion: diffusion = [x]

        for i in reversed(range(0, self.n_timesteps)):
            if self.predict_epsilon:
                x = torch.cat([catmull_spline_trajectory, is_cond, x], dim=-1)
            
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond_residual, visual_cond, timesteps)

            x = apply_conditioning(x, cond_residual, 0)

            if return_diffusion: diffusion.append(x)        

        x = catmull_spline_trajectory + x
        # Normalize the quaternions
        x[:, :, 3:7] = x[:, :, 3:7] / torch.norm(x[:, :, 3:7], dim=-1, keepdim=True)

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        elif self.get_bottleneck_feats:
            return x, self.bottleneck_feats
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, visual_cond, *args, horizon=None, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(next(iter(cond.values())))
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, visual_cond, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, spline=None, noise=None):
        x_start_noise = x_start[:, : , :-1]
        x_start_is_cond = x_start[:, :, [-1]]

        if spline is None:
            spline = torch.randn_like(x_start_noise)
        if noise is None:
            noise = torch.randn_like(x_start_noise)

        alpha = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        oneminusalpha = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        # Weighted combination of x_0 and the spline
        out = alpha * x_start_noise + oneminusalpha * noise

        # Concatenate the binary feature and the spline as the conditioning
        out = torch.cat([spline, x_start_is_cond, out], dim=-1)

        return out

    def p_losses(self, x_start, cond, t, visual_cond):
        batch_size, horizon, _ = x_start.shape
        # Extract known indices and values
        known_indices = np.array(list(cond.keys()), dtype=int)

        # candidate_no x batch_size x dim
        known_values = np.stack([c.cpu().numpy() for c in cond.values()], axis=0)
        known_values = np.moveaxis(known_values, 0, 1)

        # Sort the timepoints
        sorted_indices = np.argsort(known_indices)
        known_indices = known_indices[sorted_indices]
        known_values = known_values[:, sorted_indices]

        # Build your structured guess
        catmull_spline_trajectory = np.array([
            catmull_rom_spline_with_rotation(known_values[b, :, :-1], known_indices, horizon) 
            for b in range(batch_size)
        ])
        catmull_spline_trajectory = torch.tensor(
            catmull_spline_trajectory, 
            dtype=torch.float64, 
            device=x_start.device
        )            


        if not self.predict_epsilon:
            # Forward diffuse with the structured trajectory
            x_noisy = self.q_sample(
                x_start,
                t,
                spline=catmull_spline_trajectory,
            )
            x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

            # Reverse pass guess
            x_recon, _ = self.model(x_noisy, cond, t)
            x_recon = apply_conditioning(x_recon, cond, self.action_dim)

            # Then x_recon is the predicted x_0, compare to the true x_0
            loss, info = self.loss_fn(x_recon, x_start, cond)
        else:
            residual = x_start.clone()
            residual[:, :, :-1] -= catmull_spline_trajectory
            cond_residual = {k: torch.zeros_like(v)[:, :-1] for k, v in cond.items()}

            x_noisy = self.q_sample(
                residual,
                t,
                spline=catmull_spline_trajectory,
            )
            x_noisy = apply_conditioning(x_noisy, cond_residual, self.action_dim)

            # Reverse pass guess
            x_recon, _ = self.model(x_noisy, cond, visual_cond, t)
            x_recon = apply_conditioning(x_recon, cond_residual, 0)

            # x_recon + catmull_spline_trajectory is the predicted x_0
            loss, info = self.loss_fn(x_recon + catmull_spline_trajectory, x_start[:, :, :-1], cond)

        return loss, info

    def loss(self, x, cond, visual_cond=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, visual_cond)

    def forward(self, cond, *args, visual_cond=None, **kwargs):
        return self.conditional_sample(cond=cond, visual_cond=visual_cond, *args, **kwargs)