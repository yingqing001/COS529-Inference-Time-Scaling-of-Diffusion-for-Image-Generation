#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --output=output_slurm/%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=40:30:00
#SBATCH --account rfai



module purge
source activate rqvae_3.11

seed=0

output_dir=res/res_0727
# prompt=("rabbit", "elephant", "giraffe", "monkey")
# prompt=("rabbit", "elephant")
prompt=("cat")

# reward_fn=aesthetic
# reward_fn=incompress
reward_fn=compress

t_start=15
t_end=50

num_images=1
num_prompts=50

active_size=1
branch_size=4

# python inference.py --save_images --num_prompts $num_prompts --reward_fn $reward_fn  --t_start $t_start --t_end $t_end --active_size $active_size --branch_size $branch_size  --prompt "${prompt[@]}" --num_images $num_images --seed $seed --output_dir $output_dir


# for active_size in 8; do
#     for branch_size in 8; do
#         python inference.py --save_images --num_prompts $num_prompts --reward_fn $reward_fn  --t_start $t_start --t_end $t_end --active_size $active_size --branch_size $branch_size  --prompt "${prompt[@]}" --num_images $num_images --seed $seed --output_dir $output_dir
#     done
# done


# SD
search_method=TreeG-SD


branch_size=4

pred_xstart_scale=0.25
n_iter=2

guidance_rate=10.0

# python inference.py --dsg --guidance_rate $guidance_rate --pred_xstart_scale $pred_xstart_scale --n_iter $n_iter --search_method $search_method  --save_images --num_prompts $num_prompts --reward_fn $reward_fn  --t_start $t_start --t_end $t_end --active_size $active_size --branch_size $branch_size  --prompt "${prompt[@]}" --num_images $num_images --seed $seed --output_dir $output_dir

# n_iter=3
# for branch_size in 4 3; do
#     for pred_xstart_scale in 0.2 0.25 0.3 0.35; do
#         python inference.py --dsg --guidance_rate $guidance_rate --pred_xstart_scale $pred_xstart_scale --n_iter $n_iter --search_method $search_method  --save_images --num_prompts $num_prompts --reward_fn $reward_fn  --t_start $t_start --t_end $t_end --active_size $active_size --branch_size $branch_size  --prompt "${prompt[@]}" --num_images $num_images --seed $seed --output_dir $output_dir
#     done
# done

n_iter=2
for branch_size in 4; do
    for pred_xstart_scale in 0.25 0.3; do
        python inference.py --dsg --guidance_rate $guidance_rate --pred_xstart_scale $pred_xstart_scale --n_iter $n_iter --search_method $search_method  --save_images --num_prompts $num_prompts --reward_fn $reward_fn  --t_start $t_start --t_end $t_end --active_size $active_size --branch_size $branch_size  --prompt "${prompt[@]}" --num_images $num_images --seed $seed --output_dir $output_dir
    done
done

# n_iter=4
# for branch_size in 2; do
#     for pred_xstart_scale in 0.1 0.15 0.2 0.25 0.3; do
#         python inference.py --dsg --guidance_rate $guidance_rate --pred_xstart_scale $pred_xstart_scale --n_iter $n_iter --search_method $search_method  --save_images --num_prompts $num_prompts --reward_fn $reward_fn  --t_start $t_start --t_end $t_end --active_size $active_size --branch_size $branch_size  --prompt "${prompt[@]}" --num_images $num_images --seed $seed --output_dir $output_dir
#     done
# done






# # SC
branch_size=8
search_method=TreeG-SC
# python inference.py --search_method $search_method  --save_images --num_prompts $num_prompts --reward_fn $reward_fn  --t_start $t_start --t_end $t_end --active_size $active_size --branch_size $branch_size  --prompt "${prompt[@]}" --num_images $num_images --seed $seed --output_dir $output_dir
