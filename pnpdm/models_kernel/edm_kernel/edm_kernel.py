import torch
from guided_diffusion.models.unet_kernel_y import KernelUNet


class VPPrecondKernel(torch.nn.Module):
    def __init__(self,
        model,
        img_resolution,
        img_channels=1,
        use_fp16=False,
        beta_d=19.9,
        beta_min=0.1,
        M=1000,
        epsilon_t=1e-3,
        **kwargs
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.use_fp16 = use_fp16
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        self.model = model

    def forward(self, x, y, sigma, force_fp32=False, **model_kwargs):
        assert y is not None, "Blurred image y must be provided for KernelUNet"
        x = x.to(torch.float32).clone()
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float16 if self.use_fp16 and not force_fp32 and x.device.type == 'cuda' else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        # model_output = self.model((c_in * x).to(dtype), c_noise.flatten(), y=y)
        # Changed call to pass y as a kwarg only, not both positional and keyword
        model_output = self.model(h=(c_in * x).to(dtype), y=y, timesteps=c_noise.flatten())
        # F_x, _ = torch.split(model_output, x.shape[1], dim=1)
        F_x = model_output
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


def create_edm_from_kernel_unet(
    model_path: str,
    diffusion: dict,
    unet_cfg: dict | None = None,
    device: str | torch.device = "cuda"
):
    if unet_cfg is None:
        unet_cfg = {}

    net = KernelUNet(**unet_cfg).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    state = {k.replace("module.", ""): v for k, v in state.items()}
    # net.load_state_dict(state, strict=False)
    missing, unexp = net.load_state_dict(state, strict=False)
    print(f'[kernel] loaded ✓  missing={len(missing)}  unexpected={len(unexp)}')

    # EDM-compatible VP Preconditioned kernel denoiser
    wrapper = VPPrecondKernel(net, **diffusion).to(device)
    return wrapper
