import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from transformers.hf_argparser import HfArg, HfArgumentParser

from topomoe.inspection.feature_visualiser import (
    ImageNetVisualizer,
    LossArray,
    TotalVariation,
    ViTEnsFeatHook,
    ViTGeLUHook,
)
from topomoe.inspection.image_augmentation import (
    Clip,
    ColorJitter,
    GaussianNoise,
    Jitter,
    RepeatBatch,
    Tile,
)
from topomoe.models import create_model, list_models
from topomoe.train import Args as TrainArgs
from topomoe.utils.saving_funcs import ExperimentSaver, new_init


@dataclass
class Args:
    feature: int = HfArg(aliases=["--feature"], default=0, help="# Feature")
    layer: int = HfArg(aliases=["--layer"], default=0, help="# Layer")
    lr: float = HfArg(aliases=["--lr"], default=0.1, help="Learning Rate")
    total_variance: float = HfArg(
        aliases=["--tv"], default=1.0, help="TotalVar Lambda=v * 0.0005"
    )

    # Model
    model: str = HfArg(
        default="topomoe_tiny_2s_patch16_128",
        help=f"model ({', '.join(list_models())})",
    )
    num_classes: Optional[int] = HfArg(default=None, help="number of classes")
    num_heads: Optional[int] = HfArg(
        aliases=["--nh"], default=None, help="number of attention heads"
    )
    mlp_ratio: Optional[List[float]] = HfArg(
        aliases=["--mlpr"],
        default=None,
        help="mlp ratio. can be a single value or a list of values, "
        "e.g. --mlpr 4.0, --mlpr 4.0 4.0 4.0 2.0 2.0 2.0",
    )
    num_experts: Optional[List[int]] = HfArg(
        default=None,
        help="number of experts. can be a single value or a list of values",
    )
    mlp_conserve: Optional[bool] = HfArg(
        default=None,
        help="Divide params by num experts "
        "`expert_params = dim * mlp_ratio / num_experts`",
    )
    drop_rate: float = HfArg(aliases=["--dr"], default=0.0, help="head dropout rate")
    proj_drop_rate: float = HfArg(
        aliases=["--pdr"], default=0.0, help="projection dropout rate"
    )
    attn_drop_rate: float = HfArg(
        aliases=["--adr"], default=0.0, help="attention dropout rate"
    )
    add_pos: bool = HfArg(
        default=False, help="add stage position embedding to pooled input"
    )
    wiring_lambd: float = HfArg(
        aliases=["--wlambd"], default=0.0, help="wiring length penalty"
    )
    wiring_sigma: float = HfArg(
        aliases=["--wsigma"], default=2.0, help="wiring length radius stdev"
    )

    # Checkpoints
    checkpoint_path: Optional[Path] = HfArg(
        aliases=["--ckpt"], default=None, help="checkpoint to load"
    )
    model_args_path: Optional[Path] = HfArg(
        aliases=["--args"], default=None, help="args to load for trained model"
    )
    cuda: bool = HfArg(default=True, help="use cuda")


def main(args: Args):
    layer, feature = args.layer, args.feature
    tv = args.total_variance
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if args.checkpoint_path and args.model_args_path:
        model_arg_parser = HfArgumentParser(TrainArgs)
        (model_args,) = model_arg_parser.parse_yaml_file(yaml_file=args.model_args_path)
    else:
        model_args = args

    model = create_model(
        model_args.model,
        num_heads=model_args.num_heads,
        mlp_ratio=model_args.mlp_ratio,
        num_experts=model_args.num_experts,
        mlp_conserve=model_args.mlp_conserve,
        drop_rate=model_args.drop_rate,
        proj_drop_rate=model_args.proj_drop_rate,
        attn_drop_rate=model_args.attn_drop_rate,
        # add_pos=model_args.add_pos,
        wiring_lambd=model_args.wiring_lambd,
        wiring_sigma=model_args.wiring_sigma,
    )
    model: torch.nn.Module = model.to(device)

    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    saver = ExperimentSaver(
        f"{args.model}_TV{tv}_L{layer}_F{feature}", save_id=True, disk_saver=True
    )

    image_size = int(args.model.split("_")[-1])

    loss = LossArray()
    loss += ViTEnsFeatHook(
        ViTGeLUHook(model, sl=slice(layer, layer + 1)),
        key="high",
        feat=feature,
        coefficient=1,
    )
    loss += TotalVariation(2, image_size, coefficient=0.0005 * tv)

    pre, post = (
        torch.nn.Sequential(
            RepeatBatch(8),
            ColorJitter(8, shuffle_every=True),
            GaussianNoise(8, True, 0.5, 400),
            Tile(1),
            Jitter(),
        ),
        Clip(),
    )

    image = new_init(size=image_size, batch_size=1, device=device)
    visualizer = ImageNetVisualizer(
        loss_array=loss,
        device=device,
        saver=None,
        pre_aug=pre,
        post_aug=post,
        print_every=10,
        lr=args.lr,
        steps=400,
        save_every=100,
    )
    image.data = visualizer(image)
    saver.save(image, "final")


if __name__ == "__main__":
    args: Args
    parser = HfArgumentParser(Args)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a yaml file,
        # let's parse it to get our arguments.
        args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        (args,) = parser.parse_args_into_dataclasses()
    main(args)


[
    "stages.0.grid_embed",
    "stages.1.grid_embed",
    "stages.1.pool.grid_embed",
    "stages.1.maps.grid_embed",
    "stages.1.maps.expert_embed",
    "stages.1.blocks.0.attn.q.maps.grid_embed",
    "stages.1.blocks.0.attn.q.maps.expert_embed",
    "stages.1.blocks.0.mlp.fc1.maps.grid_embed",
    "stages.1.blocks.0.mlp.fc1.maps.expert_embed",
    "stages.1.blocks.0.mlp.fc2.maps.grid_embed",
    "stages.1.blocks.0.mlp.fc2.maps.expert_embed",
    "stages.1.blocks.1.attn.q.maps.grid_embed",
    "stages.1.blocks.1.attn.q.maps.expert_embed",
    "stages.1.blocks.1.mlp.fc1.maps.grid_embed",
    "stages.1.blocks.1.mlp.fc1.maps.expert_embed",
    "stages.1.blocks.1.mlp.fc2.maps.grid_embed",
    "stages.1.blocks.1.mlp.fc2.maps.expert_embed",
    "stages.2.grid_embed",
    "stages.2.pool.grid_embed",
    "stages.2.maps.grid_embed",
    "stages.2.maps.expert_embed",
    "stages.2.blocks.0.attn.q.maps.grid_embed",
    "stages.2.blocks.0.attn.q.maps.expert_embed",
    "stages.2.blocks.0.mlp.fc1.maps.grid_embed",
    "stages.2.blocks.0.mlp.fc1.maps.expert_embed",
    "stages.2.blocks.0.mlp.fc2.maps.grid_embed",
    "stages.2.blocks.0.mlp.fc2.maps.expert_embed",
    "stages.2.blocks.1.attn.q.maps.grid_embed",
    "stages.2.blocks.1.attn.q.maps.expert_embed",
    "stages.2.blocks.1.mlp.fc1.maps.grid_embed",
    "stages.2.blocks.1.mlp.fc1.maps.expert_embed",
    "stages.2.blocks.1.mlp.fc2.maps.grid_embed",
    "stages.2.blocks.1.mlp.fc2.maps.expert_embed",
]
[
    "stages.0.pos_embed",
    "stages.1.pos_embed",
    "stages.1.pool.pos_embed",
    "stages.1.pool.weight",
    "stages.1.maps.pos_embed",
    "stages.1.maps.weight",
    "stages.1.blocks.0.attn.q.maps.pos_embed",
    "stages.1.blocks.0.attn.q.maps.weight",
    "stages.1.blocks.0.mlp.fc1.maps.pos_embed",
    "stages.1.blocks.0.mlp.fc1.maps.weight",
    "stages.1.blocks.0.mlp.fc2.maps.pos_embed",
    "stages.1.blocks.0.mlp.fc2.maps.weight",
    "stages.1.blocks.1.attn.q.maps.pos_embed",
    "stages.1.blocks.1.attn.q.maps.weight",
    "stages.1.blocks.1.mlp.fc1.maps.pos_embed",
    "stages.1.blocks.1.mlp.fc1.maps.weight",
    "stages.1.blocks.1.mlp.fc2.maps.pos_embed",
    "stages.1.blocks.1.mlp.fc2.maps.weight",
    "stages.2.pos_embed",
    "stages.2.pool.pos_embed",
    "stages.2.pool.weight",
    "stages.2.maps.pos_embed",
    "stages.2.maps.weight",
    "stages.2.blocks.0.attn.q.maps.pos_embed",
    "stages.2.blocks.0.attn.q.maps.weight",
    "stages.2.blocks.0.mlp.fc1.maps.pos_embed",
    "stages.2.blocks.0.mlp.fc1.maps.weight",
    "stages.2.blocks.0.mlp.fc2.maps.pos_embed",
    "stages.2.blocks.0.mlp.fc2.maps.weight",
    "stages.2.blocks.1.attn.q.maps.pos_embed",
    "stages.2.blocks.1.attn.q.maps.weight",
    "stages.2.blocks.1.mlp.fc1.maps.pos_embed",
    "stages.2.blocks.1.mlp.fc1.maps.weight",
    "stages.2.blocks.1.mlp.fc2.maps.pos_embed",
    "stages.2.blocks.1.mlp.fc2.maps.weight",
]
