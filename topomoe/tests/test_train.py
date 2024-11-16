import pytest
import sys
sys.path.append('../')
from topomoe.src import train

configs = {
    "vit_small": train.Args(
        name="debug_train_vit_small",
        out_dir="topomoe/test_results",
        model="vit_small_patch16_128", 
        dataset= "hfds/clane9/imagenet-100",
        workers=1,
        batch_size=1024,
        overwrite=True,
        debug=True,
    ),
    "vit_base": train.Args(
        name="debug_train_vit_small",
        out_dir="topomoe/test_results",
        model="vit_base_patch16_128",
        dataset= "hfds/clane9/imagenet-100",
        workers=1,
        batch_size=1024,
        overwrite=True,
        debug=True,
    ),
    "transformer": train.Args(
        name="debug_train_transformer",
        out_dir="topomoe/test_results",
        model="quadmoe_tiny_1s_patch16_128",
        dataset="hfds/clane9/imagenet-100",
        workers=0,
        batch_size=1024,
        overwrite=True,
        debug=True,
    ),
    "transformer_v2": train.Args(
        name="debug_train_transformer_v2",
        out_dir="topomoe/test_results",
        model="topomoe_tiny_1s_patch16_128",
        dataset="hfds/clane9/imagenet-100",
        workers=0,
        batch_size=32,
        overwrite=True,
        debug=True,
    ),
    "quadmoe": train.Args(
        name="debug_train_quadmoe",
        out_dir="topomoe/test_results",
        model="quadmoe_tiny_2s_patch16_128",
        dataset="hfds/clane9/imagenet-100",
        workers=0,
        batch_size=32,
        overwrite=True,
        debug=True,
    ),
    "softmoe": train.Args(
        name="debug_train_softmoe",
        out_dir="topomoe/test_results",
        model="softmoe_tiny_2s_patch16_128",
        dataset="hfds/clane9/imagenet-100",
        workers=0,
        batch_size=32,
        overwrite=True,
        debug=True,
    ),
    "topomoe": train.Args(
        name="debug_train_topomoe",
        out_dir="topomoe/test_results",
        model="topomoe_tiny_2s_patch16_128",
        wiring_lambd=0.01,
        dataset="hfds/clane9/imagenet-100",
        workers=0,
        batch_size=32,
        overwrite=True,
        debug=True,
    ),
    "aug": train.Args(
        name="debug_train_aug",
        out_dir="topomoe/test_results",
        model="quadmoe_tiny_1s_patch16_128",
        dataset="hfds/clane9/imagenet-100",
        scale=[0.1, 0.3],
        ratio=[1 / 4, 4 / 1],
        hflip=0.5,
        color_jitter=0.4,
        workers=0,
        batch_size=32,
        overwrite=True,
        debug=True,
    )
}




@pytest.mark.parametrize(
    "config",
    [
        "vit_small",
        "vit_base",
        "transformer",
        "transformer_v2",
        "quadmoe",
        "softmoe",
        "topomoe",
        "aug",
    ],
)
def test_train(config: str):
    args = configs[config]
    train.main(args)

test_train("vit_small")
