import torch
from ..samplers.denoiser_edm import Denoiser_EDM


class DenoiserEDMKernel(Denoiser_EDM):
    """
     EDM 采样器，用于 kernel diffusion：
    - 继承自原始 Denoiser_EDM
    - 关键改动: forward 调用 self.net(x, sigma, y=self.y_obs)
    """

    def __init__(self, net, device,  **kwargs):
        """
        参数:
            net     : 已包装好的 VPPrecondKernel 网络
            device  : cuda / cpu
            # y_obs   : 模糊图像 (1, 3, 256, 256)，作为条件输入 --  这个就不放在init里了
            kwargs  : 传给 Denoiser_EDM 的采样配置参数
        """
        super().__init__(net, device, **kwargs)
        

    @torch.no_grad()
    def __call__(self, x_noisy, y_obs, eta):
        """
        参数:
            x_noisy : 初始噪声 (1, 1, 64, 64)
            eta     : 最小噪声级别（rho 控制，回退起点）
        """
        self.y_obs = y_obs  # 条件输入: 模糊图像
        i_start = torch.min(torch.nonzero(self.sigma(self.t_steps) < eta))
        print(f"i_start = {i_start}")
        # print(f"t_steps = {self.t_steps}")

        x_next = x_noisy * self.s(self.t_steps[i_start])
        for i, (t_cur, t_next) in enumerate(zip(self.t_steps[:-1], self.t_steps[1:])):
            
            if i < i_start:
                continue

            x_cur = x_next
            t_hat = t_cur
            x_hat = x_cur
            # print(f"i = {i}, t_cur = {t_cur}")
            
            # 去噪器: 添加 condition y_obs
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
    
    
        # #--------------------------------------------------------EDM二阶
        # for i, (t_cur, t_next) in enumerate(zip(self.t_steps[:-1], self.t_steps[1:])): # 0, ..., N-1
        #     if i < i_start:
        #         # 跳过 i_start 之前的步骤
        #         continue

        #     x_cur = x_next
            
        #     # "回退"--临时增加噪声 (noise boost) - 对应原始算法中的 S_churn 部分
        #     if self.mode == 'sde':
        #         gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
        #         t_hat = self.net.round_sigma(t_cur + gamma * t_cur)
        #         x_hat = x_cur + torch.sqrt(t_hat**2 - t_cur**2) * self.S_noise * torch.randn_like(x_cur)
        #     else:
        #         # 在 PFODE 模式下不增加额外噪声
        #         t_hat = t_cur
        #         x_hat = x_cur
            
        #     # Euler Step
        #     denoised = self.net(x_hat / self.s(t_hat), self.sigma(t_hat)).to(torch.float32)
            
        #     # 计算梯度，与原始 EDM-SDE 一致
        #     d_cur = (x_hat - self.s(t_hat) * denoised) / t_hat
            
        #     # Euler 预测步骤
        #     x_euler = x_hat + (t_next - t_hat) * d_cur
            
        #     # 应用2nd-order校正（Heun 方法）
        #     if i < self.num_steps - 1 and self.solver == 'heun':
        #         # 在下一时间步计算去噪结果
        #         denoised_next = self.net(x_euler / self.s(t_next), self.sigma(t_next)).to(torch.float32)
                
        #         # 在下一时间步计算梯度
        #         d_prime = (x_euler - self.s(t_next) * denoised_next) / t_next
                
        #         # 使用平均梯度进行修正步骤（Heun 方法）
        #         x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        #     else:
        #         # 没有二阶校正时，直接使用 Euler 结果
        #         x_next = x_euler

        # return x_next
        
