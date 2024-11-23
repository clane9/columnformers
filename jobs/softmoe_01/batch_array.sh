#!/bin/bash

#SBATCH --job-name=softmoe_01
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

JOB="softmoe_01"
# ls | xargs -I {} basename {} .yaml
NAMES=(
    01_softmoe_1s
    02_softmoe_2s
    03_softmoe_2s_nc
    04_softmoe_3s
    05_softmoe_3s_nc
)

SLURM_ARRAY_TASK_ID=1
NAME="${NAMES[SLURM_ARRAY_TASK_ID]}"
CONFIG="jobs/${JOB}/configs/${NAME}.yaml"
FULL_NAME="${JOB}/${NAME}"

python topomoe/train.py $CONFIG --name $FULL_NAME
