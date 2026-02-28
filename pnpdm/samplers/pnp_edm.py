import torch, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from .denoiser_edm import Denoiser_EDM
from .denoiser_edm_kernel import DenoiserEDMKernel
from ldm.autoencoder import VQModelTorch 
from torch.nn import functional as F

def resize_with_center_crop_or_pad(tensor_2d: torch.Tensor, input_size: tuple, output_size: tuple) -> torch.Tensor:
    """
    Resize a 2D tensor by either center-cropping or zero-padding to the desired output size.
    """
    H_in, W_in = input_size
    H_out, W_out = output_size
    assert tensor_2d.shape == (H_in, W_in), f"Input tensor shape {tensor_2d.shape} does not match provided input size {input_size}"

    # Case 1: Crop
    if H_out <= H_in and W_out <= W_in:
        top = (H_in - H_out) // 2
        left = (W_in - W_out) // 2
        return tensor_2d[top:top + H_out, left:left + W_out]

    # Case 2: Pad
    pad_top = max((H_out - H_in) // 2, 0)
    pad_bottom = max(H_out - H_in - pad_top, 0)
    pad_left = max((W_out - W_in) // 2, 0)
    pad_right = max(W_out - W_in - pad_left, 0)
    return F.pad(tensor_2d, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

def save_kernel(img, step):
    if fname is None or save_root is None: return
    ker_dir = os.path.join(save_root, 'kernels_est'); os.makedirs(ker_dir, exist_ok=True)
    plt.imsave(os.path.join(ker_dir, f'{fname}_kernel_{step}.png'),
               (img.squeeze().cpu().numpy()*255), cmap='viridis')

class PnPEDM:
    def __init__(self, config, model, kdiff, operator, noiser, device):
        self.config = config
        self.model = model  # EDM image prior model (e.g., VPPrecond / create_edm_from_unet_adm)
        self.kdiff = kdiff  # EDM kernel prior model
        self.operator = operator
        self.noiser = noiser
        self.device = device
        self.kernel_process_log = []

        self.ks = 64

        if config.mode == 'vp':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.vp_kwargs, mode='pfode')
            self.kernel_edm = DenoiserEDMKernel(kdiff, device, **config.common_kwargs, **config.vp_kwargs, mode='pfode')
        elif config.mode == 've':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.ve_kwargs, mode='pfode')
            self.kernel_edm = DenoiserEDMKernel(kdiff, device, **config.common_kwargs, **config.ve_kwargs, mode='pfode')
        elif config.mode == 'iddpm':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.iddpm_kwargs, mode='pfode')
            self.kernel_edm = DenoiserEDMKernel(kdiff, device, **config.common_kwargs, **config.ve_kwargs, mode='pfode')
        elif config.mode == 'edm':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.edm_kwargs, mode='pfode')
            self.kernel_edm = DenoiserEDMKernel(kdiff, device, **config.common_kwargs, **config.edm_kwargs, mode='pfode')
        elif config.mode == 'vp_sde':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.vp_kwargs, mode='sde')
            self.kernel_edm = DenoiserEDMKernel(kdiff, device, **config.common_kwargs, **config.vp_kwargs, mode='sde')
        elif config.mode == 've_sde':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.ve_kwargs, mode='sde')
            self.kernel_edm = DenoiserEDMKernel(kdiff, device, **config.common_kwargs,  **config.ve_kwargs, mode='sde')
        elif config.mode == 'iddpm_sde':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.iddpm_kwargs, mode='sde')
            self.kernel_edm = DenoiserEDMKernel(kdiff, device, **config.common_kwargs, **config.iddpm_kwargs, mode='sde')
        elif config.mode == 'edm_sde':
            self.edm = Denoiser_EDM(model, device, **config.common_kwargs, **config.edm_kwargs, mode='sde')
            self.kernel_edm = DenoiserEDMKernel(kdiff, device, **config.common_kwargs,  **config.edm_kwargs, mode='sde')
        else:
            raise NotImplementedError(f"Mode {self.config.mode} is not implemented.")

    @property
    def display_name(self):
        return f'pnp-edm-{self.config.mode}-rho0={self.config.rho}-rhomin={self.config.rho_min}-kernel_rho0={self.config.kernel_rho}-kernel_rhomin={self.config.kernel_rho_min}-sigma0={self.config.sigma}-sigmamin={self.config.sigma_min} '

    def __call__(self, gt, y_n, record=False, fname=None, save_root=None, inv_transform=None, metrics={}):

        # 1) Initialization
        log = defaultdict(list)
        cmap = 'gray' if gt.shape[1] == 1 else None     
        x = self.operator.initialize(gt, y_n)

        # 2) Initial logging
        x_save = inv_transform(x)
        z_save = torch.zeros_like(x_save)
        for name, metric in metrics.items():
            log[name].append(metric(x_save, inv_transform(gt)).item())
        
        xs_save = torch.cat((inv_transform(gt), x_save), dim=-1)
        try:
            zs_save = torch.cat((inv_transform(y_n.reshape(*gt.shape)), z_save), dim=-1)
        except:
            try:
                zs_save = torch.cat((inv_transform(self.operator.A_pinv(y_n).reshape(*gt.shape)), z_save), dim=-1)
            except:
                zs_save = torch.cat((z_save, z_save), dim=-1)

        if record:
            log["gt"] = inv_transform(gt).permute(0, 2, 3, 1).squeeze().cpu().numpy()
            log["x"].append(x_save.permute(0, 2, 3, 1).squeeze().cpu().numpy())

        # 3) Sampling preparation
        samples = []    # store reconstruction samples

        iters_count_as_sample = np.linspace(
            self.config.num_burn_in_iters,
            self.config.num_iters-1,
            self.config.num_samples_per_run+1,
            dtype=int
        )[1:]
        assert self.config.num_iters-1 in iters_count_as_sample, "num_iters-1 should be included in iters_count_as_sample"
        
        sub_pbar = tqdm(range(self.config.num_iters))
        
        # construct random noise sample as blur kernel starting point
        random_noise = torch.randn((1, 1, self.ks, self.ks), device=self.device)

        # EDM initial_noise_level placeholder
        step_indices = torch.tensor([0], dtype=torch.float64, device=self.device)

        m64 = random_noise

        for i in sub_pbar:

            rho_iter = self.config.rho * (self.config.rho_decay_rate**i)
            rho_iter = max(rho_iter, self.config.rho_min)

            rho_kernel_iter = self.config.kernel_rho * (self.config.kernel_rho_decay_rate**i)  
            rho_kernel_iter = max(rho_kernel_iter, self.config.kernel_rho_min) 

            sigma_x = self.config.sigma * (self.config.sigma_decay_rate**i)
            sigma_x = max(sigma_x, self.config.sigma_min)


            fi_pred = self.kernel_edm(
                x_noisy = m64,
                y_obs = y_n,
                eta = rho_kernel_iter
            )[0]    


            fi_pred = (fi_pred+1)/2
            fi_pred = torch.clip(fi_pred, min=0.0, max=1.0)
            fi_pred = fi_pred/fi_pred.sum()

            kernel_np = fi_pred.detach().cpu().numpy().squeeze()
            log["estimated_kernel"] = kernel_np
            
            # save final kernel image into kernels_est
            if i == self.config.num_iters - 1:
                if fname is not None and save_root is not None:
                    kernel_dir = os.path.join(save_root, 'kernels_est')
                    os.makedirs(kernel_dir, exist_ok=True)
                    save_path = os.path.join(kernel_dir, f'{fname}_kernel.png')
                    plt.imsave(save_path, (fi_pred.squeeze().cpu().numpy()*255), cmap='viridis')
                    
            fi = resize_with_center_crop_or_pad(fi_pred.squeeze(), input_size=(64, 64), output_size=(256, 256))
            
            # likelihood step
            z = self.operator.proximal_generator(x, y_n, fi, sigma_x, rho_iter)

            # prior step
            x = self.edm(z, rho_iter)  
            if i in iters_count_as_sample:
                samples.append(x)
            
            m256 = self.operator.proximal_generator_kernel(fi.clone(), y_n.clone(), x.clone(), sigma_x, rho_kernel_iter)
            m64 = resize_with_center_crop_or_pad(m256.squeeze(0), output_size=(64, 64),input_size=(256, 256))


            # logging
            x_save = inv_transform(x)
            z_save = inv_transform(z)
            for name, metric in metrics.items():
                log[name].append(metric(x_save, inv_transform(gt)).item())
            sub_pbar.set_description(f'running PnP-EDM (xrange=[{x.min().item():.2f}, {x.max().item():.2f}], zrange=[{z.min().item():.2f}, {z.max().item():.2f}]) | psnr: {log["psnr"][-1]:.4f}')
            
            if i % (self.config.num_iters//10) == 0:
                xs_save = torch.cat((xs_save, x_save), dim=-1)
                zs_save = torch.cat((zs_save, z_save), dim=-1)
            
            if record:
                log["x"].append(x_save.permute(0, 2, 3, 1).squeeze().cpu().numpy())

        plt.figure(figsize=(20, 5))
        plt.subplot(1, 3, 1)
        plt.plot(log["psnr"])
        plt.title(f'psnr (max): {np.amax(log["psnr"]):.4f}, (last): {log["psnr"][-1]:.4f}')
        plt.subplot(1, 3, 2)
        plt.plot(log["ssim"])
        plt.title(f'ssim (max): {np.amax(log["ssim"]):.4f}, (last): {log["ssim"][-1]:.4f}')
        plt.subplot(1, 3, 3)
        plt.plot(log["lpips"])
        plt.title(f'lpips (min): {np.amin(log["lpips"]):.4f}, (last): {log["lpips"][-1]:.4f}')
        plt.savefig(os.path.join(save_root, 'progress', fname+"_metrics.png"))
        plt.close()

        xz_save = torch.cat((xs_save, zs_save), dim=-2).permute(0, 2, 3, 1).squeeze().cpu().numpy()
        plt.imsave(os.path.join(save_root, 'progress', fname+"_x_and_z.png"), xz_save, cmap=cmap)
        np.save(os.path.join(save_root, 'progress', fname+"_log.npy"), log)

        return torch.concat(samples, dim=0)

 
class PnPEDMBatch(PnPEDM):
    @property
    def display_name(self):
        return f'pnp-edm-batch-{self.config.mode}-rho0={self.config.rho}-rhomin={self.config.rho_min}'

    def __call__(self, gt, y_n, record=False, fname=None, save_root=None, inv_transform=None, metrics={}):
        x = torch.randn(self.config.num_samples_per_run, *gt.shape[1:]).to(gt.device)

        fi = self.kdiff(y_n) 

        nfe = []

        sub_pbar = tqdm(range(self.config.num_iters))
        for i in sub_pbar:
            rho_iter = self.config.rho * (self.config.rho_decay_rate**i)
            rho_iter = max(rho_iter, self.config.rho_min)

            # likelihood step
            z = self.operator.proximal_generator(x, y_n, fi, self.noiser.sigma, rho_iter)

            # prior step
            x = self.edm(z, rho_iter)

        return x