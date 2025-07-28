#!/bin/bash
#SBATCH --job-name=dual_blind_deblur
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out



srun torchrun --nproc_per_node=1 --master-port=29380 posterior_sample.py \
    --config config/ffhq_edm_motion_deblur256.yaml \
    --output_dir results \
    --num_runs 1 


# # Motion Deblur
# nohup torchrun --nproc_per_node=1 posterior_sample.py \
#     --config config/ffhq_edm_motion_deblur256.yaml \
#     --output_dir ./results/motion_deblur_with_kernel_gt\
#     --num_runs 1 > nohup_motion.log 2>&1 &
