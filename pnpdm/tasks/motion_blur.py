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

        if self.fixed_kernel_path is not None:
            self.generate_kernel_(fixed_kernel_path=self.fixed_kernel_path)
        elif self.kernel_bank is None:
            self.generate_kernel_(seed=seed)


    
    def _build_full_spectrum(self):
        """Build full_kernel and full_spectrum from the current spatial kernel."""
        pre1 = (self.img_dim - self.kernel.shape[0]) // 2
        post1 = self.img_dim - self.kernel.shape[0] - pre1
        pre2 = (self.img_dim - self.kernel.shape[1]) // 2
        post2 = self.img_dim - self.kernel.shape[1] - pre2
        self.full_kernel = torch.nn.functional.pad(
            self.kernel, (pre1, post1, pre2, post2), "constant", 0
        ).type(torch.complex64).to(self.device)
        self.full_spectrum = fft2(self.full_kernel)[None, None]


    def generate_kernel_(self, seed=None, fixed_kernel_path=None, index=None):

        # 1) fixed kernel from file
        if fixed_kernel_path is not None:
            try:
                ker_img = Image.open(fixed_kernel_path).convert("I")
                ker_img = ker_img.resize((self.kernel_size, self.kernel_size),
                                        resample=Image.LANCZOS)
                ker_np  = np.asarray(ker_img, dtype=np.float32)
                ker_np /= (ker_np.sum() + 1e-8)
                self.kernel = torch.tensor(ker_np, dtype=torch.float32)

                self._build_full_spectrum()
                print(f"[kernel] fixed kernel loaded from {fixed_kernel_path}")
                return
            except Exception as e:
                print(f"[kernel] load error: {e}  -> fallback to random")

        # 2) kernel from preloaded kernel_bank
        if index is not None and getattr(self, "kernel_bank", None) is not None:
            if index >= len(self.kernel_bank):
                raise IndexError(f"kernel_bank has only {len(self.kernel_bank)} kernels, index {index} is out of range")
            ker_np = self.kernel_bank[index].astype(np.float32)
            ker_np /= (ker_np.sum() + 1e-8)
            self.kernel = torch.tensor(ker_np, dtype=torch.float32)

            # save_dir = "./kernel_visuals_from_generate"  
            # os.makedirs(save_dir, exist_ok=True)
            # save_path = os.path.join(save_dir, f"{index:03d}.png")
            # plt.imsave(save_path, ker_np, cmap='hot')
            # print(f"[kernel] saved kernel #{index} to {save_path}")
            self._build_full_spectrum()

            return

        # 3) random kernel
        self._generate_random_kernel(seed)
        # build full_kernel / full_spectrum for the random kernel
        self._build_full_spectrum()
        
        # return self.kernel.cpu().numpy()
    

    def _generate_random_kernel(self, seed=None):
        """
        Generate a random motion blur kernel.

        Args:
            seed (int, optional): random seed for reproducibility.
        """
        import numpy as np
        
        if seed is not None:
            np.random.seed(seed)
        
        self.kernel = Kernel(size=(self.kernel_size, self.kernel_size), 
                            intensity=self.intensity)
        self.kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)



    @property
    def display_name(self):
        return 'mblur-circ'
    
    def forward(self, x, **kwargs):  # y = H * x
        return fftshift(ifft2(self.full_spectrum * fft2(x)).real, dim=(-2,-1))


    def transpose(self, y):  # x = H^T * y
        return fftshift(ifft2(torch.conj(self.full_spectrum) * fft2(y)).real, dim=(-2,-1))


    def proximal_generator(self, x, y, fi, sigma, rho):

        fi_spectrum = fft2(fi)  #[None, None]

        power = fi_spectrum * torch.conj(fi_spectrum) # H^TH
        inv_spectrum = 1 / ((power / sigma**2) + (torch.ones_like(x) / rho**2)) 
        noise = ifft2(torch.sqrt(inv_spectrum) * fft2(torch.randn_like(x))).real

        Y_trans=fftshift(ifft2(torch.conj(fi_spectrum ) * fft2(y/ sigma**2)).real, dim=(-2,-1))
        mu_x = ifft2(inv_spectrum * fft2(Y_trans + (x / rho**2))).real
        return mu_x + noise
    
    def proximal_generator_kernel(self, fi, y, x, sigma, rho):

        x_spectrum = torch.fft.fft2(x) # [None, None]
        power = x_spectrum * torch.conj(x_spectrum) # H^TH
        inv_spectrum = 1 / ((power / sigma**2) + (torch.ones_like(fi) / rho**2)) 
        noise = torch.fft.ifft2(torch.sqrt(inv_spectrum) * torch.fft.fft2(torch.randn_like(fi))).real
        Y_trans=fftshift(torch.fft.ifft2(torch.conj(x_spectrum ) * torch.fft.fft2(y/ sigma**2)).real, dim=(-2,-1))
        mu_m = torch.fft.ifft2(inv_spectrum * torch.fft.fft2(Y_trans + (fi / rho**2))).real

        fi_new = mu_m + noise
        fi_new = 0.2989 * fi_new[:, 0:1, :, :] + 0.5870 * fi_new[:, 1:2, :, :] + 0.1140 * fi_new[:, 2:3, :, :]
        fi_new  = fi_new[0][0] 

        return fi_new

    def proximal_for_admm(self, x, y, rho):
        power = self.full_spectrum * torch.conj(self.full_spectrum)
        inv_spectrum = 1 / (power + rho)
        return ifft2(inv_spectrum * fft2(self.transpose(y) + rho * x)).real

    def initialize(self, gt, y):
        return torch.zeros_like(gt)