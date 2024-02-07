#!/bin/bash

#SBATCH --job-name=micro-in100_01
# #SBATCH --partition=GPU-shared
#SBATCH --partition=GPU-shared,GPU-small
#SBATCH -N 1
#SBATCH --ntasks=5
#SBATCH --gpus=v100-32:1
#SBATCH --time=01:00:00
# #SBATCH --array=0-10
# #SBATCH --array=11
#SBATCH --array=12
#SBATCH --account=med230001p

# Set some environment variables
ROOT="/ocean/projects/med230001p/clane2/code/columnformers"
cd $ROOT

# Set up python environment
source .venv/bin/activate

# Setup wandb
source .env
wandb login

JOB="micro-in100_01"
NAMES=(
    "baseline"
    "untied_norm"
    "untied_attn"
    "untied_mlp"
    "untied_attn_mlp"
    "untied_all"
    "untied_all_nh1"
    "untied_all_novp"
    "untied_all_nh1_novp"
    "untied_all_bottle"
    "untied_all_nh1_novp_bottle"
    "recurrent"
    "recurrent_lowlr"
)

NAME="${NAMES[SLURM_ARRAY_TASK_ID]}"
CONFIG="jobs/${JOB}/configs/${NAME}.yaml"
FULL_NAME="${JOB}/${SLURM_ARRAY_TASK_ID}_${NAME}"

python columnformers/train.py $CONFIG --name $FULL_NAME
