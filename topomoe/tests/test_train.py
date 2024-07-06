import pytest

from topomoe import train

configs = {
    "transformer": train.Args(
        name="debug_train_transformer",
        out_dir="topomoe/test_results",
        model="quadformer_tiny_1s_patch16_128",
        dataset="hfds/clane9/imagenet-100",
        workers=0,
        batch_size=32,
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
    "quadformer": train.Args(
        name="debug_train_quadformer",
        out_dir="topomoe/test_results",
        model="quadformer_tiny_2s_patch16_128",
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
        model="quadformer_tiny_1s_patch16_128",
        dataset="hfds/clane9/imagenet-100",
        scale=[0.1, 0.3],
        ratio=[1 / 4, 4 / 1],
        hflip=0.5,
        color_jitter=0.4,
        workers=0,
        batch_size=32,
        overwrite=True,
        debug=True,
    ),
}


@pytest.mark.parametrize(
    "config",
    [
        "transformer",
        "transformer_v2",
        "quadformer",
        "topomoe",
        "aug",
    ],
)
def test_train(config: str):
    args = configs[config]
    train.main(args)
