#!/bin/bash

#SBATCH --job-name=baseline                     
#SBATCH --partition=gpuA800                     
#SBATCH --nodes=1                               
#SBATCH --ntasks=8                              
#SBATCH --gres=gpu:8                            
#SBATCH --cpus-per-task=8                       
#SBATCH --time=6:00:00                          
#SBATCH --output=./slurm_output/%x-%j.out  # Standard output
#SBATCH --error=./slurm_output/err/%x-%j.err   # Standard error

module purge
module load compiler/gcc/10.4.0-intel
module load cuda/11.3

postfix="_"
checkpoint_dir="out/"

export PYTHONUNBUFFERED=1

srun -N 1 -n 1 --gres=gpu:1 --exclusive python train.py --gin_configs "configs/llff3.gin" --postfix $postfix --checkpoint_dir $checkpoint_dir --dataset_id "fern" &
srun -N 1 -n 1 --gres=gpu:1 --exclusive python train.py --gin_configs "configs/llff3.gin" --postfix $postfix --checkpoint_dir $checkpoint_dir --dataset_id "flower" &
srun -N 1 -n 1 --gres=gpu:1 --exclusive python train.py --gin_configs "configs/llff3.gin" --postfix $postfix --checkpoint_dir $checkpoint_dir --dataset_id "horns" &
srun -N 1 -n 1 --gres=gpu:1 --exclusive python train.py --gin_configs "configs/llff3.gin" --postfix $postfix --checkpoint_dir $checkpoint_dir --dataset_id "leaves" &
srun -N 1 -n 1 --gres=gpu:1 --exclusive python train.py --gin_configs "configs/llff3.gin" --postfix $postfix --checkpoint_dir $checkpoint_dir --dataset_id "orchids" &
srun -N 1 -n 1 --gres=gpu:1 --exclusive python train.py --gin_configs "configs/llff3.gin" --postfix $postfix --checkpoint_dir $checkpoint_dir --dataset_id "room" &
srun -N 1 -n 1 --gres=gpu:1 --exclusive python train.py --gin_configs "configs/llff3.gin" --postfix $postfix --checkpoint_dir $checkpoint_dir --dataset_id "trex" &
srun -N 1 -n 1 --gres=gpu:1 --exclusive python train.py --gin_configs "configs/llff3.gin" --postfix $postfix --checkpoint_dir $checkpoint_dir --dataset_id "fortress" &
wait


srun -N 1 -n 1 --gres=gpu:1 --exclusive python eval.py --gin_configs "configs/llff3.gin" --postfix $postfix --checkpoint_dir $checkpoint_dir --dataset_id "fern" &
srun -N 1 -n 1 --gres=gpu:1 --exclusive python eval.py --gin_configs "configs/llff3.gin" --postfix $postfix --checkpoint_dir $checkpoint_dir --dataset_id "flower" &
srun -N 1 -n 1 --gres=gpu:1 --exclusive python eval.py --gin_configs "configs/llff3.gin" --postfix $postfix --checkpoint_dir $checkpoint_dir --dataset_id "horns" &
srun -N 1 -n 1 --gres=gpu:1 --exclusive python eval.py --gin_configs "configs/llff3.gin" --postfix $postfix --checkpoint_dir $checkpoint_dir --dataset_id "room" &
srun -N 1 -n 1 --gres=gpu:1 --exclusive python eval.py --gin_configs "configs/llff3.gin" --postfix $postfix --checkpoint_dir $checkpoint_dir --dataset_id "trex" &
srun -N 1 -n 1 --gres=gpu:1 --exclusive python eval.py --gin_configs "configs/llff3.gin" --postfix $postfix --checkpoint_dir $checkpoint_dir --dataset_id "fortress" &
srun -N 1 -n 1 --gres=gpu:1 --exclusive python eval.py --gin_configs "configs/llff3.gin" --postfix $postfix --checkpoint_dir $checkpoint_dir --dataset_id "leaves" &
srun -N 1 -n 1 --gres=gpu:1 --exclusive python eval.py --gin_configs "configs/llff3.gin" --postfix $postfix --checkpoint_dir $checkpoint_dir --dataset_id "orchids" &
wait