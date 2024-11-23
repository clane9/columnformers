#!/bin/bash

#SBATCH --job-name=topomoe_02
#SBATCH --partition=GPU-shared
#SBATCH -N 1
#SBATCH --ntasks=5
#SBATCH --gpus=v100-32:1
#SBATCH --time=02:00:00
#SBATCH --array=2-3
#SBATCH --account=med230001p

# Set some environment variables
ROOT="/ocean/projects/med230001p/clane2/code/columnformers"
cd $ROOT

# Set up python environment
source .venv/bin/activate

# Setup wandb
source .env
wandb login

JOB="topomoe_02"
# ls | xargs -I {} basename {} .yaml
NAMES=(
    01_topomoe_3s
    02_topomoe_3s_dyn
    03_topomoe_3s_wl-0.1
    04_topomoe_3s_dyn_wl-0.1
)

NAME="${NAMES[SLURM_ARRAY_TASK_ID]}"
CONFIG="jobs/${JOB}/configs/${NAME}.yaml"
FULL_NAME="${JOB}/${NAME}"

python topomoe/train.py $CONFIG --name $FULL_NAME
