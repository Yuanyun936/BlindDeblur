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

    Args:
        tensor_2d (torch.Tensor): A 2D tensor of shape (H_in, W_in)
        input_size (tuple): The original size (H_in, W_in)
        output_size (tuple): The target size (H_out, W_out)

    Returns:
        torch.Tensor: The resized 2D tensor with shape (H_out, W_out)
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
        self.model = model  # create_edm_from_unet_adm(**kwargs)--VPPrecond(model, **diffusion_config, **kwargs)
        self.kdiff = kdiff 
        self.operator = operator
        self.noiser = noiser
        self.device = device
        self.kernel_process_log = []
        self.ks =64 ###这里后续要改成传参进来！
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

        # 1 Initialization
        log = defaultdict(list)
        cmap = 'gray' if gt.shape[1] == 1 else None     
        x = self.operator.initialize(gt, y_n)
        
        # self.kdiff.eval()  
        # y_n_01 = (y_n + 1) / 2
        # print(f"[init] y_n_01 shape: {y_n_01.shape}, min: {y_n_01.min().item():.4f}, max: {y_n_01.max().item():.4f}")
        # y_n_01_b = y_n_01.unsqueeze(0)

        # y_n_b = y_n_linear.unsqueeze(0) # y_n, y_n_b: [-1,1]

        y_n_b = y_n

        # 2 初始log
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

        # 3 样本采集准备
        samples = []    #用于存储采样【i.e.重建】结果

       
        # 计算在哪些迭代次数保存样本：从num_burn_in_iters到num_iters-1，采样总点数为num_samples_per_run
        iters_count_as_sample = np.linspace(
            self.config.num_burn_in_iters,  # 40    起点
            self.config.num_iters-1,        # 100-1 终点
            self.config.num_samples_per_run+1, # 20+1   采样次数
            dtype=int
        )[1:]   # 从第二个元素开始取
        assert self.config.num_iters-1 in iters_count_as_sample, "num_iters-1 should be included in iters_count_as_sample"
        
        sub_pbar = tqdm(range(self.config.num_iters))
        
        # m64 = torch.randn([1,1, self.ks, self.ks], device=self.device)
        # # m64 = m64/m64.sum()

        batch = 1
        ks = self.ks
        


        # Step 2: 构造 random noise 样本（Blur kernel 起点）
        random_noise = torch.randn((batch, 1, ks, ks), device=self.device)

        # Step 3: 构造 EDM initial_noise_level（通常 step_indices=0 表示采样开始）
        step_indices = torch.tensor([0], dtype=torch.float64, device=self.device)
        # initial_noise_level = (sigma_max ** (1 / rho) +
        #                     step_indices / (num_steps - 1) *
        #                     (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

        # Step 4: 乘以初始sigma，实现初始化
        m64 = random_noise #* initial_noise_level.item()
        # Normalize to [-1, 1] range
        m64_min = m64.min()
        m64_max = m64.max()
        # m64 = 2.0 * (m64 - m64_min) / (m64_max - m64_min) - 1.0 

        # print(f"[init] m64 mean: {m64.mean().item():.8f}, min: {m64.min().item():.8f}, max: {m64.max().item():.8f}")

        for i in sub_pbar:
            rho_iter = self.config.rho * (self.config.rho_decay_rate**i)    # 逐步衰减rho
            rho_iter = max(rho_iter, self.config.rho_min)   # 限制rho的最小值

            rho_kernel_iter = self.config.kernel_rho * (self.config.kernel_rho_decay_rate**i)  
            rho_kernel_iter = max(rho_kernel_iter, self.config.kernel_rho_min) 

            sigma_x = self.config.sigma * (self.config.sigma_decay_rate**i)  # 逐步衰减sigma
            sigma_x = max(sigma_x, self.config.sigma_min)  # 限制sigma的最小值


            fi_pred = self.kernel_edm(
                x_noisy = m64,
                y_obs = y_n,  # condition: blurred image ∈[-1,1]
                eta = rho_kernel_iter # 起始 σ 由 ρ 控制
            )[0]    # (1,64,64)
            # print(f"fi_pred.shape={fi_pred.shape}") #([1, 1, 64, 64])
            # print(f"fi_pred: mean = {fi_pred.mean().item():.8f}, min = {fi_pred.min().item():.8f}, max = {fi_pred.max().item():.8f}")


            fi_pred = (fi_pred+1)/2 # 归一化到[0,1],会有负值
            fi_pred = torch.clip(fi_pred, min=0.0, max=1.0)  # 限制在[0,1]范围内
            # fi_pred = (fi_pred - fi_pred.min()) / (fi_pred.max() - fi_pred.min())  # 归一化到[0,1] 可视化用
            fi_pred = fi_pred/fi_pred.sum()
            
            # print(f"fi_pred: mean = {fi_pred.mean().item():.8f}, min = {fi_pred.min().item():.8f}, max = {fi_pred.max().item():.8f}")


            # 保存估计的kernel到日志和文件--------------------------------------
            kernel_np = fi_pred.detach().cpu().numpy().squeeze()
            log["estimated_kernel"] = kernel_np
            
            # 保存kernel图像
            if i % (self.config.num_iters//10) == 0:
                if fname is not None and save_root is not None:
                    kernel_dir = os.path.join(save_root, 'kernels_est')
                    save_path = os.path.join(kernel_dir, f'{fname}_kernel_{i}.png')
                    plt.imsave(save_path, (fi_pred.squeeze().cpu().numpy()*255), cmap='viridis')
                    # print(f'[kernel] saved → {save_path}')

                    self.kernel_process_log.append(fi_pred.squeeze().cpu().numpy())
                    
            # ----------------------------------------------------------------
            # print(f"[est] fi_pred shape: {fi_pred.shape}, mean:{fi_pred.mean().item():.8f} min: {fi_pred.min().item():.4f}, max: {fi_pred.max().item():.8f}")

            fi = resize_with_center_crop_or_pad(fi_pred.squeeze(), input_size=(64, 64), output_size=(256, 256))
            # 注：squeeze() 默认会移除所有为 1 的维度


            # likelihood step   --生成与观测数据一致的中间结果z
            # z = self.operator.proximal_generator(x, y_n, fi, self.noiser.sigma, rho_iter)
            z = self.operator.proximal_generator(x, y_n, fi, sigma_x, rho_iter)


            # prior step    --把z当做噪声，生成更新的图像估计x
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

        if fname is not None and save_root is not None and hasattr(self, 'kernel_process_log'):
            kernel_log_array = np.stack(self.kernel_process_log, axis=0)  # shape: (N, H, W)
            save_path_npy = os.path.join(save_root, 'kernels_est', f'kernel_process_{fname}.npy')
            np.save(save_path_npy, kernel_log_array)
            print(f"[kernel_process] saved full kernel evolution → {save_path_npy}")

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



        # logging
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
        # logging
        sub_pbar = tqdm(range(self.config.num_iters))
        for i in sub_pbar:
            rho_iter = self.config.rho * (self.config.rho_decay_rate**i)
            rho_iter = max(rho_iter, self.config.rho_min)

            # likelihood step
            z = self.operator.proximal_generator(x, y_n, fi, self.noiser.sigma, rho_iter)

            # prior step
            x = self.edm(z, rho_iter)

        return x