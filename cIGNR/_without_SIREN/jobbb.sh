#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --job-name=train_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=out/graph.out



module purge
module load 2025
module load Anaconda3/2025.06-1

# Your job starts in the directory where you call sbatch
cd $HOME/...


source activate $HOME/miniconda3/envs/env

srun python train.py --save_output --repr "graph"