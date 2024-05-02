import pytest

from columnformers import train

configs = {
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
    "recurrent": train.Args(
        model="vision_columnformer_r_tiny_patch16_128",
        dataset="debug-100",
        workers=0,
        batch_size=32,
        out_dir="test_results",
        name="debug_train_recurrent",
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
    "moe": train.Args(
        model="vision_moemixer_tiny_patch16_128",
        dataset="debug-100",
        moe_experts="1,1,2,2,4,4",
        wiring_lambd=0.1,
        tv_lambd=0.001,
        workers=0,
        batch_size=32,
        out_dir="test_results",
        name="debug_train_moe",
        overwrite=True,
        debug=True,
    ),
    "tut": train.Args(
        model="vision_tut_tiny_patch16_128",
        dataset="debug-100",
        moe_experts="12",
        mlp_ratio=2.0,
        wiring_lambd=0.1,
        workers=0,
        batch_size=32,
        out_dir="test_results",
        name="debug_train_tut",
        overwrite=True,
        debug=True,
    ),
}


@pytest.mark.parametrize(
    "config",
    [
        "transformer",
        "feedforward",
        "recurrent",
        "wiring",
        "moe",
        "tut",
    ],
)
def test_train(config: str):
    args = configs[config]
    train.main(args)


if __name__ == "__main__":
    pytest.main([__file__])
