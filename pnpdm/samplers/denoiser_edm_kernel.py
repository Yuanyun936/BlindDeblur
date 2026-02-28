import torch
from ..samplers.denoiser_edm import Denoiser_EDM


class DenoiserEDMKernel(Denoiser_EDM):
    """EDM sampler for kernel diffusion.

    Inherits from Denoiser_EDM and modifies the forward call to
    condition on the observed blurry image y_obs.
    """

    def __init__(self, net, device,  **kwargs):
        """Initialize the kernel EDM denoiser.

        Args:
            net:       preconditioned kernel network (e.g., VPPrecondKernel).
            device:    target device (e.g., 'cuda' or 'cpu').
            **kwargs:  additional sampler configuration passed to Denoiser_EDM.
        """
        super().__init__(net, device, **kwargs)
        

    @torch.no_grad()
    def __call__(self, x_noisy, y_obs, eta):
        """Run conditional EDM sampling for the kernel.

        Args:
            x_noisy: initial noise tensor, e.g. shape (1, 1, 64, 64).
            y_obs:   observed blurry image used as condition.
            eta:     minimum noise level (controls roll-back start via rho).
        """
        # store conditional observation
        self.y_obs = y_obs
        i_start = torch.min(torch.nonzero(self.sigma(self.t_steps) < eta))
        print(f"i_start = {i_start}")

        x_next = x_noisy * self.s(self.t_steps[i_start])
        for i, (t_cur, t_next) in enumerate(zip(self.t_steps[:-1], self.t_steps[1:])):
            
            if i < i_start:
                continue

            x_cur = x_next
            t_hat = t_cur
            x_hat = x_cur
            
            # Denoiser with condition y_obs
            denoised = self.net(x_hat / self.s(t_hat), self.y_obs, self.sigma(t_hat)).to(torch.float32)

            # Euler step
            d_cur = (x_hat - self.s(t_hat) * denoised) / t_hat
            x_euler = x_hat + (t_next - t_hat) * d_cur

            if i < self.num_steps - 1 and self.solver == "heun":
                denoised_next = self.net(x_euler / self.s(t_next), self.sigma(t_next), y=self.y_obs).to(torch.float32)
                d_prime = (x_euler - self.s(t_next) * denoised_next) / t_next
                x_next = x_hat + (t_next - t_hat) * 0.5 * (d_cur + d_prime)
            else:
                x_next = x_euler

        return x_next
        
