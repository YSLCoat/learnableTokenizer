#!/bin/bash

#SBATCH --job-name=train-model
#SBATCH --account=ec232
#SBATCH --time=35:00:00
#SBATCH --partition=ifi_accel    # To use the accelerator nodes
#SBATCH --gpus=4
#SBATCH --nodes=2
#SBATCH --mem-per-gpu=100G
#SBATCH --cpus-per-task=10

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge && \
module load Python/3.10.8-GCCcore-12.2.0 CUDA/12.1.0 && \
source /projects/ec232/venvs/g01env/bin/activate && \
pip install --user [skimage] && \
python3 src/vision_transfromer/train.py "$@"