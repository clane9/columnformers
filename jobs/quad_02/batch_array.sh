#!/bin/bash

#SBATCH --job-name=quad_02
#SBATCH --partition=GPU-shared
#SBATCH -N 1
#SBATCH --ntasks=5
#SBATCH --gpus=v100-32:1
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --account=med230001p

# Set some environment variables
ROOT="/ocean/projects/med230001p/clane2/code/columnformers"
cd $ROOT

# Set up python environment
source .venv/bin/activate

# Setup wandb
source .env
wandb login

JOB="quad_02"
NAMES=(
    01_quad_1s
    02_quad_2s
    03_quad_2s_nc
    04_quad_3s
    05_quad_3s_nc
)

NAME="${NAMES[SLURM_ARRAY_TASK_ID]}"
CONFIG="jobs/${JOB}/configs/${NAME}.yaml"
FULL_NAME="${JOB}/${NAME}"

python topomoe/train.py $CONFIG --name $FULL_NAME
