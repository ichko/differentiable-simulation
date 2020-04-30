from argparse import Namespace

hparams = Namespace(
    should_log=True,
    env_name='CubeCrash-v0',
    precondition_size=2,
    dataset_size=25000,
    frame_size=(64, 64),
    epochs=500,
    bs=32,
    log_interval=40,
    lr=0.001,
    DEVICE='cuda',
)
