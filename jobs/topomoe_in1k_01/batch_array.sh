#!/bin/bash

#SBATCH --job-name=topomoe_in1k_01
#SBATCH --partition=GPU-shared,GPU-small
#SBATCH -N 1
#SBATCH --ntasks=10
#SBATCH --gpus=v100-32:2
#SBATCH --time=01:00:00
#SBATCH --array=0
#SBATCH --account=med220004p

export OMP_NUM_THREADS=10

# Set some environment variables
ROOT="/ocean/projects/med220004p/clane2/med230001p/clane2/code/columnformers"
cd $ROOT

# Set up python environment
source .venv/bin/activate

# Setup wandb
source .env
wandb login

JOB="topomoe_in1k_01"
# ls | xargs -I {} basename {} .yaml
NAMES=(
    01_topomoe_small_3s
)

NAME="${NAMES[SLURM_ARRAY_TASK_ID]}"
CONFIG="jobs/${JOB}/configs/${NAME}.yaml"
FULL_NAME="${JOB}/${NAME}"

torchrun --standalone --nproc_per_node=2 \
    src/topomoe/train.py \
    $CONFIG \
    --name $FULL_NAME
