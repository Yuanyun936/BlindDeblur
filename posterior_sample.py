import yaml
import argparse
from pathlib import Path
import torch, logging, os
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from types import SimpleNamespace
# Parallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# PnP-DM modules
from pnpdm.data import get_dataset, get_dataloader
from pnpdm.tasks import get_operator, get_noise, MotionBlurCircular     #, RetinalBlurCircular
from pnpdm.models import get_model as get_img_model
from pnpdm.models_kernel import get_model as get_kernel_model
from pnpdm.samplers import get_sampler

# Metrics
from monai.metrics import PSNRMetric, SSIMMetric
from taming.modules.losses.lpips import LPIPS

import csv

#
from guided_diffusion.ddpm.simplified_kernel_diffusion import SimplifiedKernelDiffusion
from guided_diffusion.models.unet_kernel_y import KernelUNet

def setup_distributed():
    """设置分布式训练环境"""
    # 环境变量由 torchrun 自动设置
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # 如果 MASTER_ADDR 和 MASTER_PORT 未设置，设置默认值
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        
    return rank, world_size, local_rank


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config



def posterior_sample(config_path, output_dir=None, record=False, num_runs=1):
    # 设置分布式环境
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0

    config = load_config(config_path)
    
    config['record'] = record
    config['num_runs'] = num_runs

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if is_main_process:
        print(f"Using device: {device}")
        print(f"World size: {world_size}")

    if output_dir is None:
        output_dir = Path("./results") / Path(config_path).stem
    

    
    # prepare dataloader
    transform = transforms.Compose([
        transforms.Normalize((0.5), (0.5))
    ])
    inv_transform = transforms.Compose([
        transforms.Normalize((-1), (2)),
        transforms.Lambda(lambda x: x.clamp(0, 1).detach())
    ])

    model = get_img_model(**config['model'])    # 注：此处之后可以给kdiff也写一个函数（为了一致-“清晰”）
    # operator = get_operator(**config['task']['operator'], device=device)


    # ------------- 0. 读取 config 并抽取路径 -----------------
    op_cfg = config['task']['operator'].copy()
    kernel_test_path = op_cfg.pop('kernel_test_path', None)  # 弹掉多余键

    # ------------- 1. 如给定路径则马上加载 --------------------
    if kernel_test_path:
        kernel_bank = np.load(kernel_test_path)              # shape (100, H, W)
        op_cfg['kernel_bank'] = kernel_bank                  # 放进构造参数
    else:
        kernel_bank = None

    # ------------- 2. 创建 operator 时一并传入 ----------------
    operator = get_operator(**op_cfg, device=device)         # <-- kernel_bank 已带进去
    noiser = get_noise(**config['task']['noise'])
    dataset = get_dataset(**config['data'], transform=transform)


    if is_main_process and kernel_bank is not None:
        print(f"[kernel_bank] loaded {kernel_bank.shape} kernels from {kernel_test_path}")

    # Create distributed sampler for the dataset
    dist_sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    
    dataloader = get_dataloader(
        dataset, 
        batch_size=1, 
        num_workers=4, 
        train=False,
        sampler=dist_sampler
    )

    num_test_images = len(dataset)

    # load model---------- 1. 图像 EDM ----------
    # model = get_img_model(**vars(model_config))
    model = model.to(device)
    # Wrap model with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    model.eval()

    # ---------- 2. Kernel EDM ----------
    if 'kernel_model' in config:
        kdiff = get_kernel_model(**config['kernel_model']).to(device)
        kdiff.eval()
    else:
        kdiff = None
    

    # load sampler
    sampler_config = SimpleNamespace(**config['sampler'])
    sampler = get_sampler(sampler_config, model=model, kdiff=kdiff, operator=operator, noiser=noiser, device=device)


    # working directory
    exp_name = '_'.join([operator.display_name, noiser.display_name, sampler.display_name])
    logger = logging.getLogger(exp_name)
    out_path = os.path.join(output_dir, exp_name)
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        for img_dir in ['gt', 'meas', 'kernels_est','recon', 'progress']:
            os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    #---------------------------------------------------------------------------------------------------#
    # inference
    #---------------------------------------------------------------------------------------------------#
    ## Prepare
    meta_log = defaultdict(list)
    meta_log["statistics_based_on_one_sample"] = defaultdict(list)
    meta_log["statistics_based_on_mean"] = defaultdict(list)
    metrics = {
        'psnr': PSNRMetric(max_val=1),
        'ssim': SSIMMetric(spatial_dims=2),
        'lpips': LPIPS().to(device).eval(),
    }

    if is_main_process:
        csv_path = os.path.join(out_path, "image_metrics_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "psnr_sample_avg", "ssim_sample_avg", "lpips_sample_avg",
                "consistency_sample_avg",
                "psnr_mean_avg",  "ssim_mean_avg",  "lpips_mean_avg",
                "consistency_mean_avg"
            ])

    for i, ref_img in enumerate(dataloader):
        # print(f"Processing image {i+1}/{num_test_images} on rank {rank}...")
        # continue
        # Get global index across all processes
        global_idx = i * world_size + rank
        file_idx = f"{global_idx:05d}"
        
        logger.info(f"Process {rank}: Inference for image {global_idx} on device {device}")
        ref_img = ref_img.to(device)
        cmap = 'gray' if ref_img.shape[1] == 1 else None

        # # 只有当没有指定fixed kernel路径时才 Regenerate kernel for motion blur
        # if isinstance(operator, MotionBlurCircular):
        #     if 'fixed_kernel_path' not in config['task']['operator'] or not config['task']['operator']['fixed_kernel_path']:
        #         operator.generate_kernel_(seed=global_idx)
            # ───────── ② 为每张图像设置指定 kernel ──────────
        if isinstance(operator, MotionBlurCircular):
            if hasattr(operator, 'kernel_bank'):           # 使用预置 kernel
                operator.generate_kernel_(index=i)
                print(f"Using kernel from kernel_bank at index {i}")
            elif not config['task']['operator'].get('fixed_kernel_path'):
                operator.generate_kernel_(seed=global_idx) # 旧的随机策略


        # Forward measurement model (Ax + n)
        y_n = noiser(operator.forward(ref_img))


        ## Logging
        log = defaultdict(list)
        
        # 1. Logging and Saving Gt (For Single Image):
        log["consistency_gt"] = torch.norm(operator.forward(ref_img) - y_n).item()
        log["gt"] = inv_transform(ref_img).permute(0, 2, 3, 1).squeeze().cpu().numpy()
        
        if is_main_process or world_size == 1:
            plt.imsave(os.path.join(out_path, 'gt', file_idx+'.png'), log["gt"], cmap=cmap)
        
        # 2. Save measurements and Kernel
        try:
            log["meas"] = inv_transform(y_n.reshape(*ref_img.shape)).permute(0, 2, 3, 1).squeeze().cpu().numpy()
            if is_main_process or world_size == 1:
                plt.imsave(os.path.join(out_path, 'meas', file_idx+'.png'), log["meas"], cmap=cmap)
                if hasattr(operator, 'kernel'):
                    plt.imsave(os.path.join(out_path, 'meas', file_idx+'_kernel.png'), operator.kernel.detach().cpu())
                    kernel_np = operator.kernel.detach().cpu().numpy()                

        except:
            try:
                log["meas"] = inv_transform(y_n).permute(0, 2, 3, 1).squeeze().cpu().numpy()
                if is_main_process or world_size == 1:
                    plt.imsave(os.path.join(out_path, 'meas', file_idx+'.png'), log["meas"], cmap=cmap)
            except:
                log["meas"] = inv_transform(operator.A_pinv(y_n).reshape(*ref_img.shape)).permute(0, 2, 3, 1).squeeze().cpu().numpy()
                if is_main_process or world_size == 1:
                    plt.imsave(os.path.join(out_path, 'meas', file_idx+'_pinv.png'), log["meas"], cmap=cmap)

        ## Sampling
        for j in tqdm(range(config['num_runs'])):
            samples = sampler(
                gt=ref_img, 
                y_n=y_n, 
                record=config['record'], 
                fname=file_idx+f'_run_{j}', 
                save_root=out_path, 
                inv_transform=inv_transform, 
                metrics=metrics
            )
            samples = inv_transform(samples)
            sample = samples[[-1]]  # take the last sample as the single sample for calculating metrics
            if len(samples) > 1:
                mean, std = torch.mean(samples, dim=0, keepdim=True), torch.std(samples, dim=0, keepdim=True)

            ## Metrics calculation and Save Reconstruction Image
            
            # 1. Single sample metrics
            log["samples"].append(sample.permute(0, 2, 3, 1).squeeze().cpu().numpy())
            for name, metric in metrics.items():
                log[name+"_sample"].append(metric(sample, inv_transform(ref_img)).item())
            log["consistency_sample"].append(torch.norm(operator.forward(transform(sample)) - y_n).item())
            
            if is_main_process or world_size == 1:
                plt.imsave(os.path.join(out_path, 'recon', file_idx+f'_run_{j}_sample.png'), log["samples"][-1], cmap=cmap)
            
            # 2. Multiple samples metrics
            if len(samples) > 1:
                log["means"].append(mean.permute(0, 2, 3, 1).squeeze().cpu().numpy())
                log["stds"].append(std.permute(0, 2, 3, 1).squeeze().cpu().numpy())
                for name, metric in metrics.items():
                    log[name+"_mean"].append(metric(mean, inv_transform(ref_img)).item())
                log["consistency_mean"].append(torch.norm(operator.forward(transform(mean)) - y_n).item())
                
                if is_main_process or world_size == 1:
                    plt.imsave(os.path.join(out_path, 'recon', file_idx+f'_run_{j}_mean.png'), log["means"][-1], cmap=cmap)

        if is_main_process or world_size == 1:
            np.save(os.path.join(out_path, 'recon', file_idx+'_log.npy'), log)
            
            with open(os.path.join(out_path, 'recon', file_idx+'_metrics.txt'), "w") as f:
                f.write(f'Statistics based on ONE sample for each run ({config["num_runs"]} runs in total):\n')
                f.write('\n')
                for name, _ in metrics.items():
                    f.write(f'{name} (avg over {config["num_runs"]} runs): {np.mean(log[name+"_sample"])}\n')
                f.write(f'consistency_sample (avg over {config["num_runs"]} runs): {np.mean(log["consistency_sample"])}\n')
                f.write('\n')
                for name, _ in metrics.items():
                    best_fn = np.amin if name == 'lpips' else np.amax
                    f.write(f'{name} (best among {config["num_runs"]} runs): {best_fn(log[name+"_sample"])}\n')
                f.write(f'consistency_sample (best among {config["num_runs"]} runs): {np.amin(log["consistency_sample"])}\n')
                if len(samples) > 1:
                    f.write('\n')
                    f.write('='*70+'\n')
                    f.write('\n')
                    f.write(f'Statistics based on the mean over {len(samples)} samples for each run ({config["num_runs"]} runs in total):\n')
                    f.write('\n')
                    for name, _ in metrics.items():
                        f.write(f'{name} (avg over {config["num_runs"]} runs): {np.mean(log[name+"_mean"])}\n')
                    f.write(f'consistency_mean (avg over {config["num_runs"]} runs): {np.mean(log["consistency_mean"])}\n')
                    f.write('\n')
                    for name, _ in metrics.items():
                        best_fn = np.amin if name == 'lpips' else np.amax
                        f.write(f'{name} (best among {config["num_runs"]} runs): {best_fn(log[name+"_mean"])}\n')
                    f.write(f'consistency_mean (best among {config["num_runs"]} runs): {np.amin(log["consistency_mean"])}\n')
                f.write('\n')
                f.write('='*70+'\n')
                f.write('\n')
                f.write(f'consistency (gt): {log["consistency_gt"]}\n')
                f.close()

        psnr_s_avg  = np.mean(log["psnr_sample"])
        ssim_s_avg  = np.mean(log["ssim_sample"])
        lpips_s_avg = np.mean(log["lpips_sample"])
        cons_s_avg  = np.mean(log["consistency_sample"])

        psnr_m_avg  = np.mean(log["psnr_mean"])
        ssim_m_avg  = np.mean(log["ssim_mean"])
        lpips_m_avg = np.mean(log["lpips_mean"])
        cons_m_avg  = np.mean(log["consistency_mean"])



        if is_main_process:                 
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    psnr_s_avg, ssim_s_avg, lpips_s_avg, cons_s_avg,
                    psnr_m_avg, ssim_m_avg, lpips_m_avg, cons_m_avg
                ])


        # Meta logging (collect statistics across all images)
        meta_log["consistency_gt"].append(log["consistency_gt"])
        sample_recon_mean = torch.mean(torch.from_numpy(np.array(log["samples"])), dim=0)
        if len(sample_recon_mean.shape) == 2:
            sample_recon_mean = sample_recon_mean.unsqueeze(2)  # add a channel dimension
        sample_recon_mean = sample_recon_mean.permute(2, 0, 1).unsqueeze(0).to(device)
        for name, metric in metrics.items():
            meta_log["statistics_based_on_one_sample"][name+"_mean_recon_of_all_runs"].append(metric(sample_recon_mean, inv_transform(ref_img)).item())
            meta_log["statistics_based_on_one_sample"][name+"_last_of_all_runs"].append(log[name+"_sample"][-1])
            best_fn = np.amin if name == 'lpips' else np.amax
            meta_log["statistics_based_on_one_sample"][name+"_best_of_all_runs"].append(best_fn(log[name+"_sample"]))
        meta_log["statistics_based_on_one_sample"]["consistency_mean_recon_of_all_runs"].append(torch.norm(operator.forward(transform(sample_recon_mean)) - y_n).item())
        meta_log["statistics_based_on_one_sample"]["consistency_last_of_all_runs"].append(log["consistency_sample"][-1])
        meta_log["statistics_based_on_one_sample"]["consistency_best_of_all_runs"].append(np.amin(log["consistency_sample"]))
        
        if len(samples) > 1:
            mean_recon_mean = torch.mean(torch.from_numpy(np.array(log["means"])), dim=0)
            if len(mean_recon_mean.shape) == 2:
                mean_recon_mean = mean_recon_mean.unsqueeze(2)  # add a channel dimension
            mean_recon_mean = mean_recon_mean.permute(2, 0, 1).unsqueeze(0).to(device)
            for name, metric in metrics.items():
                meta_log["statistics_based_on_mean"][name+"_mean_recon_of_all_runs"].append(metric(mean_recon_mean, inv_transform(ref_img)).item())
                meta_log["statistics_based_on_mean"][name+"_last_of_all_runs"].append(log[name+"_mean"][-1])
                best_fn = np.amin if name == 'lpips' else np.amax
                meta_log["statistics_based_on_mean"][name+"_best_of_all_runs"].append(best_fn(log[name+"_mean"]))
            meta_log["statistics_based_on_mean"]["consistency_mean_recon_of_all_runs"].append(torch.norm(operator.forward(transform(mean_recon_mean)) - y_n).item())
            meta_log["statistics_based_on_mean"]["consistency_last_of_all_runs"].append(log["consistency_mean"][-1])
            meta_log["statistics_based_on_mean"]["consistency_best_of_all_runs"].append(np.amin(log["consistency_mean"]))

    # Synchronize processes before gathering results
    dist.barrier()
    
    # Gather all meta_log data from different processes
    if world_size > 1:
        # Convert meta_log to a format that can be gathered
        all_meta_logs = [None] * world_size if is_main_process else None
        meta_log_tensor = torch.tensor([1], device=device)  # Dummy tensor to synchronize with
        dist.barrier()  # Ensure all processes have reached this point
        
        # Gather all meta_logs on rank 0
        dist.gather(meta_log_tensor, all_meta_logs if is_main_process else None, dst=0)
        
        if is_main_process:
            # Merge all meta_logs (simplified - in practice would need to serialize and deserialize)
            # This is a placeholder for the actual gathering mechanism
            pass

    # Save meta log (only on main process)
    if is_main_process:
        np.save(os.path.join(out_path, 'meta_log.npy'), meta_log)
        
        with open(os.path.join(out_path, 'meta_metrics.txt'), "w") as f:
            f.write(f'Statistics based on ONE sample for each run ({config["num_runs"]} runs in total) of each test image:\n')
            f.write('\n')
            for name, _ in metrics.items():
                f.write(f'{name}_mean_recon_of_{config["num_runs"]}_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_one_sample"][name+"_mean_recon_of_all_runs"])}\n')
            f.write(f'consistency_mean_recon_of_{config["num_runs"]}_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_one_sample"]["consistency_mean_recon_of_all_runs"])}\n')
            f.write('\n')
            for name, _ in metrics.items():
                f.write(f'{name}_last_of_{config["num_runs"]}_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_one_sample"][name+"_last_of_all_runs"])}\n')
            f.write(f'consistency_last_of_all_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_one_sample"]["consistency_last_of_all_runs"])}\n')
            f.write('\n')
            for name, _ in metrics.items():
                f.write(f'{name}_best_of_{config["num_runs"]}_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_one_sample"][name+"_best_of_all_runs"])}\n')
            f.write(f'consistency_best_of_all_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_one_sample"]["consistency_best_of_all_runs"])}\n')
            if len(samples) > 1:
                f.write('\n')
                f.write('='*70+'\n')
                f.write('\n')
                f.write(f'Statistics based on the mean over {len(samples)} samples for each run ({config["num_runs"]} runs in total) of each test image:\n')
                f.write('\n')
                for name, _ in metrics.items():
                    f.write(f'{name}_mean_recon_of_{config["num_runs"]}_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_mean"][name+"_mean_recon_of_all_runs"])}\n')
                f.write(f'consistency_mean_recon_of_{config["num_runs"]}_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_mean"]["consistency_mean_recon_of_all_runs"])}\n')
                f.write('\n')
                for name, _ in metrics.items():
                    f.write(f'{name}_last_of_{config["num_runs"]}_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_mean"][name+"_last_of_all_runs"])}\n')
                f.write(f'consistency_last_of_all_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_mean"]["consistency_last_of_all_runs"])}\n')
                f.write('\n')
                for name, _ in metrics.items():
                    f.write(f'{name}_best_of_{config["num_runs"]}_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_mean"][name+"_best_of_all_runs"])}\n')
                f.write(f'consistency_best_of_all_runs (avg over {num_test_images} test images): {np.mean(meta_log["statistics_based_on_mean"]["consistency_best_of_all_runs"])}\n')
            f.write('\n')
            f.write('='*70+'\n')
            f.write('\n')
            f.write(f'consistency (gt) (avg over {num_test_images} test images): {np.mean(meta_log["consistency_gt"])}\n')
            f.close()

    
    
    
    dist.barrier()  # Ensure all processes wait before cleaning up
    dist.destroy_process_group()
    
    logger.info(f"Finished inference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run posterior sampling with a specified config")
    parser.add_argument("--config", type=str, default="config/eye_edm_motion_deblur256.yaml", 
                        help="Path to the config file")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Directory to save results (defaults to ./results/{config_name})")
    parser.add_argument("--record", action="store_true", help="Whether to record all intermediate samples")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of sampling runs per image")
    
    args = parser.parse_args()


    posterior_sample(
        config_path=args.config,
        output_dir=args.output_dir,
        record=args.record,
        num_runs=args.num_runs
    )