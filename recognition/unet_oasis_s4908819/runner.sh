#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=unet_oasis_test
#SBATCH -o unet_oasis_test.out

source ~/miniconda3/bin/activate
conda activate torch
cd ~/recognition/unet_oasis_s4908819

python train.py
