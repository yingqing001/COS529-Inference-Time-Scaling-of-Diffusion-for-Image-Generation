import os
import argparse
import torch
import pickle
import random
import numpy as np

from model import TreeSearchStableDiffusionPipeline, DDIMCustomScheduler

import sys
sys.path.append("rewards")
from rewards.rewards import aesthetic_score, jpeg_incompressibility, jpeg_compressibility

def get_args_parser():
    parser = argparse.ArgumentParser(description="Tree Search Stable Diffusion")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Path to the pretrained model or identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--prompt",
        nargs='+',
        default="A fantasy landscape with mountains and a river",
        help="The prompt to guide the image generation.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="The number of images to generate.",
    )
    parser.add_argument(
        "--active_size",
        type=int,
        default=1,
        help="The number of active nodes in the tree search.",
    )
    parser.add_argument(
        "--branch_size",
        type=int,
        default=1,
        help="The number of branches to explore in the tree search.",
    )
    parser.add_argument(
        "--t_start",
        type=int,
        default=0,
        help="The starting timestep for the diffusion process.",
    )
    parser.add_argument(
        "--t_end",
        type=int,
        default=50,
        help="The ending timestep for the diffusion process.",
    )
    parser.add_argument(
        "--reward_fn",
        type=str,
        default="aesthetic",
        help="The reward function to use for the tree search.",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Whether to save the generated images.",
    )
    parser.add_argument(
        "--imagenet_class",
        action="store_true",
        help="Whether to save the generated images.",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=1,
        help="The number of times to repeat the image generation.",
    )
    parser.add_argument(
        "--search_method",
        type=str,
        default="TreeG-SC",
        help="The search method to use for the tree search.",
    )
    parser.add_argument(
        "--pred_xstart_scale",
        type=float,
        default=1.0,
        help="The scale for the predicted xstart in TreeG-SD.",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="The number of iterations for the TreeG-SD search method.",        
    )
    parser.add_argument(
        "--dsg",
        action="store_true",
        help="Whether to use the DSG (Dynamic Search Graph) method in TreeG-SD.",
    )
    parser.add_argument(
        "--guidance_rate",
        type=float,
        default=1.0,
        help="The guidance rate for the diffusion process.",
    )
 

    return parser


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Seed: {seed}")
    print(f"Device: {device}")


    pipeline = TreeSearchStableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
    )

    ## set reward function
    def test_fn(x):
        reward = x.mean(dim=tuple(range(1, x.ndim)))
        return reward
    
    if args.reward_fn == "incompress":
    # reward_fn = aesthetic_score()
        reward_fn = jpeg_incompressibility()
    elif args.reward_fn == "compress":
        reward_fn = jpeg_compressibility()
    elif args.reward_fn == "aesthetic":
        reward_fn = aesthetic_score()
    else:
        raise ValueError(f"Unknown reward function: {args.reward_fn}")
    
    pipeline.set_reward_fn(reward_fn)

    
    pipeline = pipeline.to(device)
    pipeline.enable_attention_slicing()  # lower memory footprint (optional)

    os.makedirs(args.output_dir, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(seed)

    print("Search Method:", args.search_method)
    SD_kwargs = {}
    if args.search_method == "TreeG-SD":
        SD_kwargs = {
            "pred_xstart_scale": args.pred_xstart_scale,
            "n_iter": args.n_iter,
            "dsg": args.dsg,
            "guidance_rate": args.guidance_rate,
        }
        print("Hyperparameters for TreeG-SD:\n", SD_kwargs)

    print(f"active size {args.active_size} and branch size {args.branch_size}")
    # print(f"Prompt: {args.prompt}")

    with open("assets/imagenet_classes.txt", "r") as file:
        imagenet_classes = file.readlines()
    imagenet_classes = [class_name.strip() for class_name in imagenet_classes]
    random.seed(args.seed)
    prompts = random.sample(imagenet_classes, args.num_prompts)

    rewards = []
    images_all = []

    print(f"Guidance step: {args.t_start} to {args.t_end}")

    print(f"Prompts: {prompts}")
    
    for i, prompt in enumerate(prompts):
        print(f"Prompt: {prompt}")
        result, image_pt = pipeline(
            search_method=args.search_method,
            active_size=args.active_size,
            branch_size=args.branch_size,
            t_start=args.t_start,
            t_end=args.t_end,
            prompt=prompt,
            num_images_per_prompt=5,
            num_inference_steps=50,
            generator=generator,
            eta=1.0,
            SD_kwargs=SD_kwargs,
        )
        images = result.images  # List[ PIL.Image.Image ]

        # print(images)

        for image in images:
            images_all.append(image)
        
        ### test rewards
        reward = reward_fn(image_pt)
        # print(f"Reward: {reward}")

        
        print(f"Prompt: {prompt}, Reward: {reward.mean().item()}")
        rewards.append(torch.tensor(reward))



    dirs = os.path.join(args.output_dir, args.reward_fn, args.search_method, f"seed_{args.seed}_active_{args.active_size}_branch_{args.branch_size}")
    if args.search_method == "TreeG-SD":
        dirs = os.path.join(dirs, f"pred_xstart_scale_{args.pred_xstart_scale}_n_iter_{args.n_iter}_guidance_rate_{args.guidance_rate}")
    # os.makedirs(dirs, exist_ok=True)

    # print("rewards:", rewards)
    # print("images_all:", len(images_all))

    # rewards are list of torch.Tensor (B,), to get (NB,)
    rewards = torch.cat(rewards).cpu().numpy()
    print(f"Shape of rewards: {rewards.shape}")

    ## get mean and std
    mean = np.mean(rewards)
    std = np.std(rewards)
    dirs += f"_mean_{mean:.3f}"
    os.makedirs(dirs, exist_ok=True)
    print("------ Reward Statistics ------")
    print(f"Mean: {mean}, Std: {std}")
    print("\n")
    ## save rewards
    stats = (args.active_size, args.branch_size, mean.item(), std.item())
    with open(os.path.join(dirs, "stats.pkl"), "wb") as f:
        pickle.dump(stats, f)
    print(f"Saved stats to {os.path.join(dirs, 'stats.pkl')}")
    ## save meta rewards
    with open(os.path.join(dirs, "rewards.pkl"), "wb") as f:
        pickle.dump(rewards, f)


    if args.save_images:
        images_dir = os.path.join(dirs, "images")
        os.makedirs(images_dir, exist_ok=True)
        print(f"Saving images to {images_dir}")
        for idx, image in enumerate(images_all):
            # print(idx)
            file_path = os.path.join(images_dir, f"image_{idx:03d}_reward_{rewards[idx].item():.3f}.png")
            image.save(file_path)
            # print(f"Saved {file_path}")



if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
