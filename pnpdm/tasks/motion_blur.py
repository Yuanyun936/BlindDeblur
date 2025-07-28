import torch
import torch.nn as nn
import numpy as np
from torch.fft import fft2, ifft2, fftshift
from . import register_operator, LinearOperator
from .kernel import Kernel
from PIL import Image
import cv2
from guided_diffusion.util.utils_torch import conv_kernel
import matplotlib.pyplot as plt
import os

@register_operator(name='motion_blur_circ')
class MotionBlurCircular(LinearOperator):
    def __init__(self, kernel_size, intensity, channels, img_dim, device, seed=None, fixed_kernel_path=None,kernel_bank=None) -> None:
        assert channels in [1, 3], 'The number of channels should be either 1 or 3!'
        self.kernel_size = kernel_size
        self.intensity = intensity
        self.img_dim = img_dim
        self.device = device
        self.fixed_kernel_path = fixed_kernel_path
        self.kernel_bank = kernel_bank
        self.kernel_process_log = [] 
        self.generate_kernel_(seed, fixed_kernel_path)


    
    '''    def generate_kernel_(self, seed=None, fixed_kernel_path=None):
        if fixed_kernel_path is not None:
        # 使用预定义的核
            try:
                # 加载图像->灰度图
                kernel_img = Image.open(fixed_kernel_path).convert("I")
                kernel_img = kernel_img.resize((self.kernel_size, self.kernel_size), 
                                            resample=Image.LANCZOS)
                # 转换为numpy数组
                fixed_kernel = np.array(kernel_img, dtype=np.float32)
                # norm
                fixed_kernel = fixed_kernel / fixed_kernel.sum()
                self.kernel = torch.tensor(fixed_kernel, dtype=torch.float32)
                print(f"Fixed kernel loaded from {fixed_kernel_path}")
            except Exception as e:
                print(f"Error loading fixed kernel: {e}")
                print("Falling back to random kernel generation")
                # 出错时回退到随机生成
                self._generate_random_kernel(seed)
            
        else:
            # 使用随机生成的Kernel
            self._generate_random_kernel(seed)'''


    def generate_kernel_(self, seed=None, fixed_kernel_path=None, index=None):
        """
        优先级：
        1. fixed_kernel_path → 从图像加载
        2. index            → 从 self.kernel_bank[index] 取
        3. seed             → 随机生成（旧逻辑）
        """
        # ---------- 1. 固定路径 ----------
        if fixed_kernel_path is not None:
            try:
                ker_img = Image.open(fixed_kernel_path).convert("I")
                ker_img = ker_img.resize((self.kernel_size, self.kernel_size),
                                        resample=Image.LANCZOS)
                ker_np  = np.asarray(ker_img, dtype=np.float32)
                ker_np /= (ker_np.sum() + 1e-8)
                self.kernel = torch.tensor(ker_np, dtype=torch.float32)
                print(f"[kernel] fixed kernel loaded from {fixed_kernel_path}")
                return
            except Exception as e:
                print(f"[kernel] load error: {e}  -> fallback to random")

        # ---------- 2. 预置 kernel_bank ----------
        if index is not None and hasattr(self, "kernel_bank"):
            if index >= len(self.kernel_bank):
                raise IndexError(f"kernel_bank 只有 {len(self.kernel_bank)} 个，索引 {index} 越界")
            ker_np = self.kernel_bank[index].astype(np.float32)
            ker_np /= (ker_np.sum() + 1e-8)
            self.kernel = torch.tensor(ker_np, dtype=torch.float32)

                #保存 kernel 为图像
            save_dir = "./kernel_visuals_from_generate"  # 自定义保存路径
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{index:03d}.png")
            plt.imsave(save_path, ker_np, cmap='hot')
            print(f"[kernel] saved kernel #{index} to {save_path}")


            # print(f"kernel形状: {self.kernel.shape}, 值范围: [{self.kernel.min()}, {self.kernel.max()}], 均值: {self.kernel.mean()}")
            # self.kernel = self.kernel-min(self.kernel.flatten())
            
            # self._generate_random_kernel(seed)
            # print(f"随机生成的kernel形状: {self.kernel.shape}, 值范围: [{self.kernel.min()}, {self.kernel.max()}], 均值: {self.kernel.mean()}")

            # 用零填充Kernel以匹配图像尺寸
            pre1 = (self.img_dim-self.kernel.shape[0])//2
            post1 = self.img_dim-self.kernel.shape[0]-pre1
            pre2 = (self.img_dim-self.kernel.shape[1])//2
            post2 = self.img_dim-self.kernel.shape[1]-pre2
            self.full_kernel = torch.nn.functional.pad(
                self.kernel, (pre1, post1, pre2, post2), "constant", 0
            ).type(torch.complex64).to(self.device)
            # 计算核的fft（频谱）-方便快速卷积
            self.full_spectrum = fft2(self.full_kernel)[None, None]

            return

        # ---------- 3. 随机生成 ----------
        self._generate_random_kernel(seed)
        


        # return self.kernel.cpu().numpy()
    

    def _generate_random_kernel(self, seed=None):
        """
        生成随机模糊核
        
        Arguments:
            seed {int} -- 随机种子（可选）

        """
        import numpy as np
        
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
        
        self.kernel = Kernel(size=(self.kernel_size, self.kernel_size), 
                            intensity=self.intensity)
        self.kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)



    @property
    def display_name(self):
        return 'mblur-circ'
    
    # 注1：.real 用来提取逆傅里叶变换结果的实部。由于计算过程可能产生复数值，
    #   但在大多数图像处理中，最终的图像应该是实数，因此此处取实部。
    # 注2：.fftshift 用来调整频谱的零频率分量到频谱的中心位置。
    #   （因为在傅里叶变换后，频谱的零频率分量通常位于频谱的四个角落）
    #   dim=(-2, -1) ：对二维数组的最后两个维度（H,W）进行移位。

    
    def forward(self, x, **kwargs): # y = H * x --卷积
        return fftshift(ifft2(self.full_spectrum * fft2(x)).real, dim=(-2,-1))


    def transpose(self, y): # x = H^T * y --逆卷积
        return fftshift(ifft2(torch.conj(self.full_spectrum) * fft2(y)).real, dim=(-2,-1))


    # mu_x :mu_pi 图像的估计值（生成的信号）
    # inv_spectrum： sigma_pi^2
    # full_spectrum :卷积核的频谱 H(f)
    # sigma :sigma_y
    # rho : ρ
    def proximal_generator(self, x, y, fi, sigma, rho):

        # sigma = 1
        # fi_spectrum = self.full_spectrum    # Original kernel spectrum in PnPdm
        fi_spectrum = fft2(fi)  #[None, None]

        power = fi_spectrum * torch.conj(fi_spectrum) # H^TH
        inv_spectrum = 1 / ((power / sigma**2) + (torch.ones_like(x) / rho**2)) 
        noise = ifft2(torch.sqrt(inv_spectrum) * fft2(torch.randn_like(x))).real

        Y_trans=fftshift(ifft2(torch.conj(fi_spectrum ) * fft2(y/ sigma**2)).real, dim=(-2,-1))
        mu_x = ifft2(inv_spectrum * fft2(Y_trans + (x / rho**2))).real
        # mu_x = ifft2(inv_spectrum * fft2(self.transpose(y / sigma**2) + (x / rho**2))).real # Original kernel
        return mu_x + noise
    
    def proximal_generator_kernel(self, fi, y, x, sigma, rho):
        # Convert RGB to grayscale for x and y
        # x = 0.2989 * x[:, 0:1, :, :] + 0.5870 * x[:, 1:2, :, :] + 0.1140 * x[:, 2:3, :, :]
        # y = 0.2989 * y[:, 0:1, :, :] + 0.5870 * y[:, 1:2, :, :] + 0.1140 * y[:, 2:3, :, :]
        # # fi = 0.2989 * fi[:, 0:1, :, :] + 0.5870 * fi[:, 1:2, :, :] + 0.1140 * fi[:, 2:3, :, :]        
        # print(f"x.shape: {x.shape}, y.shape: {y.shape}, fi.shape: {fi.shape}")

        # x = x[0][0]
        # y = y[0][0]
        # fi = fi[0][0]
        # fi = torch.zeros_like(x) 

        x_spectrum = torch.fft.fft2(x) # [None, None]
        power = x_spectrum * torch.conj(x_spectrum) # H^TH
        inv_spectrum = 1 / ((power / sigma**2) + (torch.ones_like(fi) / rho**2)) 
        noise = torch.fft.ifft2(torch.sqrt(inv_spectrum) * torch.fft.fft2(torch.randn_like(fi))).real
        Y_trans=fftshift(torch.fft.ifft2(torch.conj(x_spectrum ) * torch.fft.fft2(y/ sigma**2)).real, dim=(-2,-1))
        mu_m = torch.fft.ifft2(inv_spectrum * torch.fft.fft2(Y_trans + (fi / rho**2))).real
        # print(f"mu_m.shape: {mu_m.shape}, mu_m.min: {mu_m.min()}, mu_m.max: {mu_m.max()}, mu_m.mean: {mu_m.mean()}")

        fi_new = mu_m + noise
        fi_new = 0.2989 * fi_new[:, 0:1, :, :] + 0.5870 * fi_new[:, 1:2, :, :] + 0.1140 * fi_new[:, 2:3, :, :]
        fi_new  = fi_new[0][0] 
        # print(f"fi_new.shape: {fi_new.shape}, fi_new.min: {fi_new.min()}, fi_new.max: {fi_new.max()}, fi_new.mean: {fi_new.mean()}")

        return fi_new

    def proximal_for_admm(self, x, y, rho):
        power = self.full_spectrum * torch.conj(self.full_spectrum)
        inv_spectrum = 1 / (power + rho)
        return ifft2(inv_spectrum * fft2(self.transpose(y) + rho * x)).real

    def initialize(self, gt, y):
        return torch.zeros_like(gt)