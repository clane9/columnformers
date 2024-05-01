#!/bin/bash

#SBATCH --job-name=moe_01
#SBATCH --partition=GPU-shared
#SBATCH -N 1
#SBATCH --ntasks=5
#SBATCH --gpus=v100-32:1
#SBATCH --time=03:00:00
# #SBATCH --array=0-5
# #SBATCH --array=6-7
# #SBATCH --array=8-9
# #SBATCH --array=10-11
#SBATCH --array=12-13
#SBATCH --account=med230001p

# Set some environment variables
ROOT="/ocean/projects/med230001p/clane2/code/columnformers"
cd $ROOT

# Set up python environment
source .venv/bin/activate

# Setup wandb
source .env
wandb login

JOB="moe_01"
CONFIG="jobs/${JOB}/base_config.yaml"
OPTS=(
    "--moe_experts 1"
    "--moe_experts 2"
    "--moe_experts 4"
    "--moe_experts 1,1,1,2,2,2"
    "--moe_experts 1,1,1,4,4,4"
    "--moe_experts 1,1,2,2,4,4"
    "--moe_experts 2,2,2,1,1,1"
    "--moe_experts 4,4,4,1,1,1"
    "--moe_experts 1 --mlp_ratio 4,4,4,2,2,2"
    "--moe_experts 1 --mlp_ratio 4,4,4,1,1,1"
    "--moe_experts 1 --mlp_ratio 2"
    "--moe_experts 1 --mlp_ratio 1"
    "--moe_experts 1,1,1,2,2,2 --no_moe_conserve"
    "--moe_experts 1,1,1,4,4,4 --no_moe_conserve"
)
NAMES=(
    "01_E-1"
    "02_E-2"
    "03_E-4"
    "04_E-1-2"
    "05_E-1-4"
    "06_E-1-2-4"
    "07_E-2-1"
    "08_E-4-1"
    "09_E-1_R-4-2"
    "10_E-1_R-4-1"
    "11_E-1_R-2"
    "12_E-1_R-1"
    "13_E-1-2_NC"
    "14_E-1-4_NC"
)

OPT="${OPTS[SLURM_ARRAY_TASK_ID]}"
NAME="${NAMES[SLURM_ARRAY_TASK_ID]}"
FULL_NAME="${JOB}/${NAME}"

python columnformers/train.py $CONFIG --name $FULL_NAME $OPT
