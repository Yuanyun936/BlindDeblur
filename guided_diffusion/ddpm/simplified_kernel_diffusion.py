import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from collections import namedtuple

from guided_diffusion.ddpm.gaussian_diffusion import GaussianDiffusion
from guided_diffusion.ddpm.gaussian_diffusion import default, extract, identity, exists

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class SimplifiedKernelDiffusion(GaussianDiffusion):

    def __init__(
        self,
        model,  # KernelUNet
        image_size = 256, 
        train_loss = 'l2',
    ):
        super().__init__(model=model, image_size=image_size)
        
        if hasattr(model, "module"):
            self.ks = model.module.kernel_size
            self.model_channels = model.module.channels
            self.model_out_channels = model.module.out_channels
            self.model_self_condition = model.module.self_condition
        else:
            self.ks = model.kernel_size
            self.model_channels = model.channels
            self.model_out_channels = model.out_channels
            self.model_self_condition = model.self_condition

        if train_loss == 'l1':
            self.train_loss = F.l1_loss
        else:
            self.train_loss = F.mse_loss
        
        self.sample = self.p_sample_loop  

    def unnormalize_kernel(self, k):

        k_clip = torch.clip(self.unnormalize(k), 0, np.inf)
        k_out = torch.div(k_clip, torch.sum(k_clip, (1,2,3), keepdim=True)) 
        return k_out

    @torch.no_grad()
    def p_mean_variance(self, x, y, t, x_self_cond=None, clip_denoised=True):

        preds = self.model_predictions(x, y, t)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start
     
    @torch.no_grad()
    def p_sample(self, x, y, t, x_self_cond=None):

        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, y=y, t=batched_times, x_self_cond=x_self_cond, clip_denoised=True
        )

        noise = torch.randn_like(x) if t > 0 else 0.
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        
        return pred_img, x_start

    def p_sample_loop(self, y):

        device = self.betas.device
        batch = y.shape[0]
        y = self.normalize(y)

        img = torch.randn([batch, 1, self.ks, self.ks], device=device)
        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step'):
            self_cond = x_start if self.model_self_condition else None
            img, x_start = self.p_sample(img, y, t, self_cond)

        return self.unnormalize_kernel(img)

    def model_predictions(self, x, y, t, x_self_cond=None, clip_x_start=False):

        if self.model_channels == self.model_out_channels:
            # predict x_start
            pred = self.model(x, y, t)
            x_start = pred
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        else:
            # predict noise
            pred = self.model(x, y, t)
            pred_noise = pred
            x_start = self.predict_start_from_noise(x, t, pred_noise)

        if clip_x_start:
            x_start.clamp_(-1., 1.)

        return ModelPrediction(pred_noise, x_start)

    def p_losses(self, x_start, y, t, noise=None):

        noise = default(noise, lambda: torch.randn_like(x_start))

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        model_out = self.model(x_t, y, t)
        
        if self.model_channels == self.model_out_channels:
            target = x_start
        else:
            target = noise

        loss = self.train_loss(model_out, target)
        
        return loss