#!/bin/bash

#SBATCH --job-name=in100_01
#SBATCH --partition=GPU-shared
#SBATCH -N 1
#SBATCH --ntasks=5
#SBATCH --gpus=v100-32:1
#SBATCH --time=04:00:00
# #SBATCH --array=0
# #SBATCH --array=1-12
# #SBATCH --array=13-17
# #SBATCH --array=18
#SBATCH --array=19
#SBATCH --account=med230001p

# Set some environment variables
ROOT="/ocean/projects/med230001p/clane2/code/columnformers"
cd $ROOT

# Set up python environment
source .venv/bin/activate

# Setup wandb
source .env
wandb login

JOB="in100_01"
NAMES=(
    "00_baseline"
    "01_untied_norm"
    "02_untied_attn"
    "03_untied_mlp"
    "04_untied_attn_mlp"
    "05_untied_all"
    "06_untied_all_nh1"
    "07_untied_all_novp"
    "08_untied_all_nh1_novp"
    "09_untied_all_bottle"
    "10_untied_all_nh1_novp_bottle"
    "11_recurrent"
    "12_recurrent_lowlr"
    "13_untied_all_lowerlr"
    "14_untied_all_nh1_novp_lowerlr"
    "15_untied_all_bottle_lowerlr"
    "16_untied_all_nh1_novp_bottle_lowerlr"
    "17_recurrent_lowerlr"
    "18_baseline_wl0.01"
    "19_baseline_wl0.1"
)

NAME="${NAMES[SLURM_ARRAY_TASK_ID]}"
CONFIG="jobs/${JOB}/configs/${NAME}.yaml"
FULL_NAME="${JOB}/${NAME}"

python columnformers/train.py $CONFIG --name $FULL_NAME
