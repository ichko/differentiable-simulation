from argparse import Namespace

from data import generate_data
import models
import utils

import torch
from tqdm import tqdm
import wandb
import gym

import matplotlib.pyplot as plt


def fit(model, dataloader, haprams):
    model.optim_init(lr=hparams.lr)
    model = model.to(hparams.DEVICE)

    for e_id in tqdm(range(hparams.epochs)):
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            actions, preconditions, futures = [
                t.to(hparams.DEVICE) for t in batch
            ]

            loss, info = model.optim_step([[actions, preconditions], futures])

            if hparams.should_log:
                loss_log_interval = 5
                if i % loss_log_interval == 0:
                    wandb.log({
                        'loss': loss,
                        'epoch': e_id,
                    })

                if i % haprams.log_interval == 0:
                    num_log_images = 10
                    y = info['y'][:num_log_images]
                    y_pred = info['y_pred'][:num_log_images]
                    diff = abs(y - y_pred)

                    wandb.log({
                        'y': [wandb.Image(i) for i in y],
                        'y_pred': [wandb.Image(i) for i in y_pred],
                        'diff': [wandb.Image(i) for i in diff]
                    })

                    model.persist()


if __name__ == '__main__':
    models.sanity_check()

    hparams = Namespace(
        should_log=True,
        env_name='CubeCrash-v0',
        precondition_size=2,
        dataset_size=25000,
        frame_size=(32, 32),
        epochs=500,
        bs=128,
        log_interval=40,
        lr=0.0003,
        DEVICE='cuda',
    )

    env = gym.make(hparams.env_name)
    data = utils.persist(
        lambda: generate_data(
            env,
            lambda _: env.action_space.sample(),
            dataset_size=hparams.dataset_size,
            frame_size=hparams.frame_size,
            precondition_size=hparams.precondition_size,
        ),
        f'.data/{hparams.env_name}_{hparams.frame_size}_{hparams.dataset_size}.pkl',
        override=True,
    )

    model = models.ForwardModel(
        action_output_channels=32,
        precondition_channels=hparams.precondition_size * 3,
        precondition_out_channels=32,
    )
    persisted_model_name = f'.models/{hparams.env_name}.pkl'

    if hparams.should_log:
        wandb.init(project='forward_model', config=hparams)
        wandb.watch(model)
        wandb.save('data.py')
        wandb.save('models.py')
        wandb.save('torch_utils.py')
        wandb.save('utils.py')
        wandb.save('run_experiment.py')
        # persisted_model_name = f'.models/{wandb.run.name}.pkl'

    model.make_persisted(persisted_model_name)
    # if model.can_be_preloaded():
    #     print('> Preloading model')
    #     model.preload_weights()

    fit(model, dataloader, hparams)
    model.persist()
