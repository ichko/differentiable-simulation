import io
import time
import random
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange
import PIL
import cv2

import sneks
import gym

import torch
import torch.nn as nn

plt.style.use('default')

DEVICE = 'cuda'


class DQN(nn.Module):
    def __init__(self, obs_size, num_actions):
        super(DQN, self).__init__()
        hidden_dim = 512

        def dense(i, o, a=nn.Sigmoid):
            l = nn.Linear(i, o)
            return nn.Sequential(l, a())

        # def lam(func):
        #     class Lambda(nn.Module):
        #         def forward(self, *args):
        #             return func(*args)

        #     return Lambda()

        def make_dqn():
            return nn.Sequential(
                nn.Flatten(),
                dense(obs_size, hidden_dim, nn.ReLU),
                dense(hidden_dim, hidden_dim, nn.ReLU),
                nn.Linear(hidden_dim, num_actions),
            ).to(DEVICE)

        self.eval_net = make_dqn()
        self.target_net = make_dqn()

    def forward(self, obs):
        return self.eval_net(obs)

    def get_max_action(self, obs):
        obs = obs[np.newaxis, ...] / 255
        obs = torch.FloatTensor(obs).to(DEVICE)
        with torch.no_grad():
            q_vals = self.eval_net(obs)[0]

        return q_vals.argmax().cpu().detach().numpy()

    def replace_target(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    # https://github.com/cyoon1729/deep-Q-networks/blob/master/vanillaDQN/dqn.py#L51
    def loss(self, i, batch):
        discount = 0.99
        criterion = nn.MSELoss()

        if i % 100:
            self.replace_target()

        obs, actions, rewards, next_obs = batch
        obs = torch.FloatTensor(obs).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_obs = torch.FloatTensor(next_obs).to(DEVICE)

        q_vals = self(obs).index_select(dim=1, index=actions)[0]

        next_q_vals = self.target_net(next_obs).max(dim=1)[0]
        target_q_vals = next_q_vals * discount + rewards

        return criterion(q_vals, target_q_vals)


def get_experience_generator(env, model, bs):
    max_rollout_steps = 200
    buffer_size = 100_000
    randomness = 1
    randomness_decay = 0.001
    randomness_min = 0.05

    episode_rewards = []
    experience_pool = deque(maxlen=buffer_size)

    while True:
        randomness = max(randomness_min, randomness - randomness_decay)

        obs = env.reset()
        done = False
        step = 0
        episode_rewards.append(0)

        while not done and step < max_rollout_steps:
            step += 1

            use_model = random.random() > randomness
            if use_model:
                action = model.get_max_action(obs)
            else:
                action = env.action_space.sample()

            next_obs, reward, done, _info = env.step(action)
            experience_pool.append((obs / 255, action, reward, next_obs / 255))

            obs = next_obs
            episode_rewards[-1] += reward

            if len(experience_pool) >= bs:
                batch = random.sample(experience_pool, bs)
                yield [np.array(t) for t in zip(*batch)], episode_rewards


def dqn_optimize(env, model, its, next_batch):
    tr = trange(its, bar_format="{bar}{l_bar}{r_bar}")

    optimizer = torch.optim.Adam(
        params=model.eval_net.parameters(),
        lr=0.001,
    )

    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    plt.ion()
    plt.show()

    losses = []
    loss_avgs = []
    episode_ids = []

    for i in tr:
        episode_ids.append(i)
        batch, episode_rewards = next(next_batch)

        optimizer.zero_grad()
        loss = model.loss(i, batch)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        loss_avgs.append(np.mean(losses[-20:]))

        description = [
            'it %i/%i' % (i + 1, its),
            'loss: %.6f' % loss.item(),
            'reward moving avg: %.2f' % np.mean(episode_rewards[-50:]),
            ' #### ',
        ]

        axs[0].clear()
        axs[1].clear()

        axs[0].plot(episode_ids[-500:], losses[-500:], linewidth=1)
        axs[0].plot(episode_ids[-500:], loss_avgs[-500:], linewidth=2)
        axs[1].plot(episode_rewards[:-1], linewidth=2)

        fig.canvas.draw()
        fig.canvas.flush_events()

        tr.set_description(' | '.join(description))


def play_env(env, controller):
    obs = env.reset()
    while True:
        action = controller(obs)
        obs, reward, done, _info = env.step(action)
        env.render()
        print(reward)
        if done: break


if __name__ == '__main__':
    print(torch.__version__, torch.cuda.is_available())

    # env = gym.make('snek-rgb-16-v1')
    env = gym.make('CartPole-v1')
    play_env(env, lambda obs: env.action_space.sample())
    print(env.action_space, env.observation_space, env.reward_range)

    dqn = DQN(obs_size=4, num_actions=2)
    dqn_optimize(
        env=env,
        model=dqn,
        its=3000,
        next_batch=get_experience_generator(
            env,
            dqn,
            bs=32,
        ),
    )