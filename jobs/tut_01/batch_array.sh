#!/bin/bash

#SBATCH --job-name=tut_01
#SBATCH --partition=GPU-shared
#SBATCH -N 1
#SBATCH --ntasks=5
#SBATCH --gpus=v100-32:1
#SBATCH --time=08:00:00
#SBATCH --array=5-7
#SBATCH --account=med230001p

# Set some environment variables
ROOT="/ocean/projects/med230001p/clane2/code/columnformers"
cd $ROOT

# Set up python environment
source .venv/bin/activate

# Setup wandb
source .env
wandb login

JOB="tut_01"
NAMES=(
    "01_base"
    "02_ff_skip"
    "03_ff_skip_v2"
    "04_ff_skip_v3"
    "05_t_embed"
    "06_ff_skip_v4"
    "07_moe_attn"
    "08_rvit"
)

NAME="${NAMES[SLURM_ARRAY_TASK_ID]}"
CONFIG="jobs/${JOB}/configs/${NAME}.yaml"
FULL_NAME="${JOB}/${NAME}"

python columnformers/train.py $CONFIG --name $FULL_NAME
