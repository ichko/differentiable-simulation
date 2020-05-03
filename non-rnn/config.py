from argparse import Namespace
import sneks

# CubeCrash-v0
# snek-rgb-16-v1

hparams = Namespace(
    should_log=True,
    env_name='CubeCrash-v0',
    precondition_size=1,
    dataset_size=50000,
    frame_size=(64, 64),
    epochs=500,
    bs=64,
    log_interval=40,
    lr=0.001,
    DEVICE='cuda',
)
