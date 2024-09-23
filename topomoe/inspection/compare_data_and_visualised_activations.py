import h5py
import numpy as np
import torch
from PIL import Image
from timm.data import create_dataset
from torchvision import transforms
from transformers.hf_argparser import HfArgumentParser

from topomoe.inspection.features import FeatureExtractor
from topomoe.models import create_model
from topomoe.train import Args as TrainArgs

model_args_path = "topomoe/topomoe_01/10_topomoe_3s_wl-0.1/args.yaml"
checkpoint_path = "topomoe/topomoe_01/10_topomoe_3s_wl-0.1/checkpoints/ckpt-best.pt"
device = "cpu"
layer_feat_img = {
    "stages.0.blocks.0.mlp.act": {7: 4928, 83: 2502, 546: 390, 81: 2078, 1354: 474},
    "stages.0.blocks.1.mlp.act": {0: 3368, 67: 596, 356: 837, 739: 2923, 1230: 1809},
    "stages.1.blocks.0.mlp.act": {2: 4135, 31: 146, 100: 4015, 263: 2764, 305: 897},
    "stages.1.blocks.1.mlp.act": {3: 3348, 54: 3704, 125: 3769, 260: 3678, 371: 336},
    "stages.2.blocks.0.mlp.act": {1: 269, 25: 3772, 46: 1251, 78: 4104, 93: 2443},
    "stages.2.blocks.1.mlp.act": {5: 3798, 17: 3895, 31: 2988, 57: 711, 83: 2265},
}
layer_to_level = {
    "stages.0.blocks.0.mlp.act": 0,
    "stages.0.blocks.1.mlp.act": 1,
    "stages.1.blocks.0.mlp.act": 2,
    "stages.1.blocks.1.mlp.act": 3,
    "stages.2.blocks.0.mlp.act": 4,
    "stages.2.blocks.1.mlp.act": 5,
}

features_path = "topomoe_features/topomoe_tiny_3s_patch16_128/validation_features.h5"
dataset_path = "hfds/clane9/imagenet-100"

dataset = create_dataset(dataset_path, root=None, download=True)

model_arg_parser = HfArgumentParser(TrainArgs)
(model_args,) = model_arg_parser.parse_yaml_file(yaml_file=model_args_path)

model = create_model(
    model_args.model,
    num_heads=model_args.num_heads,
    mlp_ratio=model_args.mlp_ratio,
    num_experts=model_args.num_experts,
    mlp_conserve=model_args.mlp_conserve,
    drop_rate=model_args.drop_rate,
    proj_drop_rate=model_args.proj_drop_rate,
    attn_drop_rate=model_args.attn_drop_rate,
    wiring_lambd=model_args.wiring_lambd,
    wiring_sigma=model_args.wiring_sigma,
)
model: torch.nn.Module = model.to(device)

ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt["model"])

transform = transforms.ToTensor()

for layer, feats_to_img in layer_feat_img.items():
    level = layer_to_level[layer]
    for feat, img_idx in feats_to_img.items():
        try:
            img = Image.open(
                f"desktop/topomoe_tiny_3s_patch16_128_TV1.0_L{level}_F{feat}/png_ImageSaver/0_final_500_steps_lr1.0.png"
            ).copy()

            img_tensor = transform(img).unsqueeze(0)

            extractor = FeatureExtractor(model, [layer])

            extractor.model.eval()

            _, batch_features = extractor(img_tensor)

            print(f"Layer {layer} | feat {feat}")

            for name, values in batch_features.items():
                print("visualised img activation sum: ", torch.sum(values[:, :, feat]))

            with h5py.File(features_path, "r") as f:
                print(
                    "validation img activation sum: ",
                    torch.sum(torch.tensor(np.array(f[layer]))[img_idx, :, feat]),
                )

            print("-" * 20)
        except:
            pass
