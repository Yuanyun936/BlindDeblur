
# Motion Deblur
torchrun --nproc_per_node=1 posterior_sample.py \
    --config config/ffhq_edm_motion_deblur256.yaml \
    --output_dir ./results/\
    --num_runs 1



# srun torchrun --nproc_per_node=1 --master-port=29380 posterior_sample.py \
#     --config config/ffhq_edm_motion_deblur256.yaml \
#     --output_dir results \
#     --num_runs 1 

