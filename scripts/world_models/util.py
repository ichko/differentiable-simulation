import os
import random
from collections import defaultdict, deque

import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn


def optimize(model, next_batch, its, optim_args, on_it_end):
    plotter = Plotter()
    tr = trange(its)
    fit = model.optimizer(**optim_args)

    for i in tr:
        batch = next(next_batch)
        info = fit(batch)

        tr.set_description(' > Loss: %012.6f' % info['loss'])

        on_it_end(i)

        plotter.log(**info)
        if i % (its // 10) == 0:
            plotter.plot()


def play_env(env, agent, duration, fps=30):
    time_step = 0
    while True:
        obs = env.reset()
        done = False

        while not done:
            time_step += 1
            action = agent(obs)
            obs, _reward, done, _info = env.step(action)
            env.render('human')

            if time_step >= duration:
                return


def make_persisted_model(model, path):
    model.persist = lambda: torch.save(model.state_dict(), path)

    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))

    return model


class Plotter:
    def __init__(self, rows=1):
        self.rows = rows
        self.history = defaultdict(lambda: [])

    def log(self, **info):
        for name, value in sorted(info.items()):
            if type(value) is list:
                self.history[name] = value
            else:
                self.history[name].append(value)

    def plot(self):
        num_axs = len(self.history)
        cols = round(num_axs / self.rows)

        if not hasattr(self, 'fig'):
            self.fig, self.axs = plt.subplots(
                self.rows,
                cols,
                figsize=(cols * 4, self.rows * 3),
            )
            self.fig.tight_layout()
            plt.ion()
            plt.show()

            # Wrap if its 1x1 plot
            if type(self.axs) is not np.ndarray:
                self.axs = [self.axs]

            # Flatten if its NxM ploy
            if type(self.axs[0]) is np.ndarray:
                self.axs = [a for ax in self.axs for a in ax]

        for ax, (name, _value) in zip(self.axs, sorted(self.history.items())):
            ax.clear()
            ax.plot(self.history[name], linewidth=2)
            ax.set_title(name)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def get_experience_generator(
    env,
    agent,
    bs,
    randomness=1,
    randomness_min=0.01,
    randomness_decay=0.9993,
    buffer_size=100_000,
    max_rollout_steps=500,
):
    episode_rewards = []
    randomness_list = []
    experience_pool = deque(maxlen=buffer_size)

    while True:
        obs = env.reset()
        done = False
        step = 0
        episode_rewards.append(0)

        while not done and step < max_rollout_steps:
            randomness = max(randomness_min, randomness * randomness_decay)
            randomness_list.append(randomness)
            step += 1

            use_model = random.uniform(0, 1) > randomness
            if use_model:
                action = agent(obs)
            else:
                action = env.action_space.sample()

            next_obs, reward, done, _info = env.step(action)
            experience_pool.append((obs, action, reward, next_obs, done))
            obs = next_obs
            episode_rewards[-1] += reward

            if len(experience_pool) >= bs:
                batch = random.sample(experience_pool, bs)
                batch = [np.array(t) for t in zip(*batch)]
                yield batch, episode_rewards, randomness_list
