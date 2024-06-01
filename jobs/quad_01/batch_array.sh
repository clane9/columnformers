#!/bin/bash

#SBATCH --job-name=quad_01
#SBATCH --partition=GPU-shared
#SBATCH -N 1
#SBATCH --ntasks=5
#SBATCH --gpus=v100-32:1
#SBATCH --time=02:00:00
#SBATCH --array=5-6
#SBATCH --account=med230001p

# Set some environment variables
ROOT="/ocean/projects/med230001p/clane2/code/columnformers"
cd $ROOT

# Set up python environment
source .venv/bin/activate

# Setup wandb
source .env
wandb login

JOB="quad_01"
NAMES=(
    01_base
    02_branch-1
    03_branch-1_nc
    04_branch-2
    05_branch-2_nc
    06_branch-1_sharekv
    07_branch-2_sharekv
)

NAME="${NAMES[SLURM_ARRAY_TASK_ID]}"
CONFIG="jobs/${JOB}/configs/${NAME}.yaml"
FULL_NAME="${JOB}/${NAME}"

python columnformers/train.py $CONFIG --name $FULL_NAME
