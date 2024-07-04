import pytest

from topomoe import train

configs = {
    "transformer": train.Args(
        name="debug_train_transformer",
        out_dir="topomoe/test_results",
        model="vision_transformer_tiny_patch16_128",
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
}


@pytest.mark.parametrize(
    "config",
    [
        "transformer",
        "quadformer",
        "topomoe",
    ],
)
def test_train(config: str):
    args = configs[config]
    train.main(args)
