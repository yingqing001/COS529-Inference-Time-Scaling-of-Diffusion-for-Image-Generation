#!/bin/bash
#SBATCH --job-name=tts_         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=04:59:00          # total run time limit (HH:MM:SS)
#SBATCH --output=slurm_logs/%j.out
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=fail
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=yg6736@princeton.edu
#SBATCH --mem-per-cpu=50G         # memory per cpu-core (4G is default)
#SBATCH --partition=pli
#SBATCH --account=ai2_design


module purge
module load anaconda3/2024.10  
source activate rnadiff

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


branch_size=8
pred_xstart_scale=0.25
n_iter=1

guidance_rate=10.0
python inference.py --dsg --guidance_rate $guidance_rate --pred_xstart_scale $pred_xstart_scale --n_iter $n_iter --search_method $search_method  --save_images --num_prompts $num_prompts --reward_fn $reward_fn  --t_start $t_start --t_end $t_end --active_size $active_size --branch_size $branch_size  --prompt "${prompt[@]}" --num_images $num_images --seed $seed --output_dir $output_dir




# # SC
branch_size=6
search_method=TreeG-SC
# python inference.py --search_method $search_method  --save_images --num_prompts $num_prompts --reward_fn $reward_fn  --t_start $t_start --t_end $t_end --active_size $active_size --branch_size $branch_size  --prompt "${prompt[@]}" --num_images $num_images --seed $seed --output_dir $output_dir
