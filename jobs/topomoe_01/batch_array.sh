#!/bin/bash

#SBATCH --job-name=topomoe_01
#SBATCH --partition=GPU-shared
#SBATCH -N 1
#SBATCH --ntasks=5
#SBATCH --gpus=v100-32:1
#SBATCH --time=02:00:00
#SBATCH --array=10-11
#SBATCH --account=med230001p

# Set some environment variables
ROOT="/ocean/projects/med230001p/clane2/code/columnformers"
cd $ROOT

# Set up python environment
source .venv/bin/activate

# Setup wandb
source .env
wandb login

JOB="topomoe_01"
# ls | xargs -I {} basename {} .yaml
NAMES=(
    01_topomoe_1s
    02_topomoe_2s
    03_topomoe_2s_nc
    04_topomoe_3s
    05_topomoe_3s_nc
    06_topomoe_2s_np
    07_topomoe_3s_np
    08_topomoe_2s_wl-0.01
    09_topomoe_2s_wl-0.1
    10_topomoe_3s_wl-0.1
    11_topomoe_2s_wp_wl-0.1
    12_topomoe_3s_wp_wl-0.1
)

NAME="${NAMES[SLURM_ARRAY_TASK_ID]}"
CONFIG="jobs/${JOB}/configs/${NAME}.yaml"
FULL_NAME="${JOB}/${NAME}"

python topomoe/train.py $CONFIG --name $FULL_NAME
