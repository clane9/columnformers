import pytest

from columnformers import train

configs = {
    "default": train.Args(
        dataset="debug-100",
        workers=0,
        batch_size=32,
        out_dir="test_results",
        name="debug_train_default",
        overwrite=True,
        debug=True,
    ),
    "transformer": train.Args(
        model="vision_transformer_tiny_patch16_128",
        dataset="debug-100",
        workers=0,
        batch_size=32,
        out_dir="test_results",
        name="debug_train_transformer",
        overwrite=True,
        debug=True,
    ),
    "feedforward": train.Args(
        model="vision_columnformer_ff_tiny_patch16_128",
        dataset="debug-100",
        workers=0,
        batch_size=32,
        out_dir="test_results",
        name="debug_train_feedforward",
        overwrite=True,
        debug=True,
    ),
    "wiring": train.Args(
        model="vision_transformer_tiny_patch16_128",
        dataset="debug-100",
        wiring_lambd=0.1,
        workers=0,
        batch_size=32,
        out_dir="test_results",
        name="debug_train_wiring",
        overwrite=True,
        debug=True,
    ),
    "selection_lowrank": train.Args(
        model="vision_columnformer_ff_tiny_patch16_128",
        attn_mode="selection",
        mlp_rank=8,
        dataset="debug-100",
        workers=0,
        batch_size=32,
        out_dir="test_results",
        name="debug_train_selection_lowrank",
        overwrite=True,
        debug=True,
    ),
}


@pytest.mark.parametrize(
    "config",
    [
        "default",
        "transformer",
        "feedforward",
        "wiring",
        "selection_lowrank",
    ],
)
def test_train(config: str):
    args = configs[config]
    train.main(args)


if __name__ == "__main__":
    pytest.main([__file__])
