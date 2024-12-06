#!/bin/bash

#SBATCH --job-name=topomoe_in1k_02
#SBATCH --partition=GPU-shared
#SBATCH -N 1
#SBATCH --ntasks=10
#SBATCH --gpus=v100-32:1
#SBATCH --time=1-00:00:00
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

JOB="topomoe_in1k_02"
# ls | xargs -I {} basename {} .yaml
NAMES=(
    01_vit_small
    02_topomoe_small_3s_wl-0.01
    03_topomoe_small_3s_wl-0.001
    04_topomoe_small_3s_dyn_wl-0.001
)

NAME="${NAMES[SLURM_ARRAY_TASK_ID]}"
CONFIG="jobs/${JOB}/configs/${NAME}.yaml"
FULL_NAME="${JOB}/${NAME}"

torchrun --standalone --nproc_per_node=1 \
    src/topomoe/train.py \
    $CONFIG \
    --name $FULL_NAME
