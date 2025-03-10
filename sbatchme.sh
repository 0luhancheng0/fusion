#!/bin/bash
#SBATCH --job-name=fusion-experiments
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
module load mamba
mamba activate rapids-24.12
cd /home/lcheng/oz318/fusion
python fusion.py run-experiments
