#!/bin/bash

#SBATCH --job-name=train-model
#SBATCH --account=ec232
#SBATCH --time=48:00:00
#SBATCH --partition=ifi_accel    # To use the accelerator nodes
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=30G
#SBATCH --cpus-per-task=10

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge && \
module load Python/3.10.8-GCCcore-12.2.0 CUDA/12.1.0 && \
source /projects/ec232/venvs/g01env/bin/activate && \
python3 src/train.py "$@"