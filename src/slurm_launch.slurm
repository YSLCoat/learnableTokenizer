#!/usr/bin/bash

## SLURM SETUP
#SBATCH --account=ec232
#SBATCH --job-name=LearnableTokenizerViT
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=ifi_accel
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=out/job_%j.out

# Load modules and venv
module purge
module load Python/3.10.4-GCCcore-11.3.0
source /projects/ec232/venvs/g01env/bin/activate

# Set MASTER_ADDR and MASTER_PORT
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# Calculate the total number of tasks
#export WORLD_SIZE=$(($SLRUM_JOB_NUM_NODES*$SLURM_NTASKS_PER_NODE))
#export WORLD_SIZE=$(($SLURM_NTASKS * $SLURM_NPROCS))

# Print some info
echo -e "=====\nWhen: $(date)\nWhere: $(hostname) @ $(pwd)\n=====\n"
echo "Nodelist =" $SLURM_JOB_NODELIST
echo "Total number of nodes =" $SLURM_JOB_NUM_NODES
echo "NTASKS_PER_NODE ="  $SLURM_NTASKS_PER_NODE

echo "GPUS_ON_NODE =" $SLURM_GPUS_ON_NODE
echo "MASTER_PORT =" $MASTER_PORT
#echo "WORLD_SIZE =" $WORLD_SIZE
echo "MASTER_ADDR =" $MASTER_ADDR
echo "SLURM_JOB_ID = " $SLURM_JOB_ID

# Run train script
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    src/train_quix.py \
    --cfgfile src/T16_fox.toml