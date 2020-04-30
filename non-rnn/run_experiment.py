from argparse import Namespace

from config import hparams
from data import generate_data
import models.frame_transform as models
import utils

import torch
from tqdm import tqdm
import wandb
import gym

import matplotlib.pyplot as plt


def get_model():
    model = models.ForwardGym(
        num_actions=ENV.action_space.n,
        action_output_channels=32,
        precondition_channels=hparams.precondition_size * 3,
        precondition_out_channels=128,
    )
    persisted_model_name = f'.models/{hparams.env_name}.pkl'

    model.make_persisted(persisted_model_name)
    if model.can_be_preloaded():
        print('> Preloading model')
        try:
            model.preload_weights()
        except Exception as e:
            print(f'ERROR: Could not preload weights [{repr(e)}]')

    number_of_parameters = model.count_parameters()
    print(f'Number of trainable parameters: {number_of_parameters}')

    return model


ENV = gym.make(hparams.env_name)
MODEL = get_model()


def fit(model, data, haprams):
    model.optim_init(lr=hparams.lr)
    model = model.to(hparams.DEVICE)
    dataloader = model.to_dataloader(data, hparams.bs)

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
                        'lr_scheduler': model.scheduler.get_lr()[0],
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

        # End of epoch
        model.scheduler.step()

        # Log videos
        vid_paths = []
        for i in range(2):
            vid_path = f'./.videos/vid_epoch_{e_id:04}_{hparams.env_name}_{i}.webm'
            utils.produce_video(vid_path, model, ENV, haprams)
            vid_paths.append(vid_path)

        wandb.log({'example video': [wandb.Video(v) for v in vid_paths]})


if __name__ == '__main__':
    models.sanity_check()

    data = utils.persist(
        lambda: generate_data(
            ENV,
            lambda _: ENV.action_space.sample(),
            dataset_size=hparams.dataset_size,
            frame_size=hparams.frame_size,
            precondition_size=hparams.precondition_size,
        ),
        f'.data/{hparams.env_name}_{hparams.frame_size}_{hparams.dataset_size}.pkl',
        override=False,
    )

    if hparams.should_log:
        wandb.init(project='forward_model', config=hparams)
        wandb.watch(MODEL)
        wandb.save('data.py')
        wandb.save('models/*')
        wandb.save('utils.py')
        wandb.save('run_experiment.py')

    fit(MODEL, data, hparams)
    MODEL.persist()
