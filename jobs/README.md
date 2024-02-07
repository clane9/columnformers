# Experiment job files

This directory contains YAML config files and other related files used for our experiments.

The options in the YAML config files are the same as the command-line options in the [training script](../columnformers/train.py). For example, here are config options to train a baseline transformer on ImageNet-100:

```yaml
model: vision_transformer_tiny_patch16_128
desc: "baseline ViT"

dataset: imagenet-100
crop_min_scale: 0.33334
hflip: 0.5
color_jitter: 0.4
epochs: 50
amp: true
wandb: true
figure_interval: 5
```

To run a training job from the project root directory:

```bash
# Set up python environment
source .venv/bin/activate

# Setup wandb
# .env should export your WANDB_API_KEY
source .env
wandb login

# Job name and basename for config
JOB=
NAME=

# Config path and full training run name
CONFIG="jobs/${JOB}/configs/${NAME}.yaml"
FULL_NAME="${JOB}/${NAME}"

# Launch single gpu training
# For multi-gpu use e.g. torchrun
python columnformers/train.py $CONFIG --name $FULL_NAME
```

See also the SLURM scripts in the job directories for examples of running on a cluster. (Cluster details will vary.)

## Experiments

- [micro-in100_01](micro-in100_01): Overfitting experiment on Micro-ImageNet-100
  - date: 1/21/2024
  - commit: 5184c96833fc73c8280ce029aee0cb207902bcfb
- [in100_01](in100_01): Training sweep of models on ImageNet-100 with varying weight untying, parameter saving, and recurrence.
  - date: 1/22/2024
  - commit: 5184c96833fc73c8280ce029aee0cb207902bcfb
- [attn_comp_01](attn_comp_01): Training feedforward fully untied models with 3 different attention mechanisms.
  - date: 2/6/2024
  - commit: c452abafe6cb7aa53e271a1c3f364db6034a4af4
