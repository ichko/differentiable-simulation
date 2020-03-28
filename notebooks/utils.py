import IPython

import os
import io
import time
import random
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange
import PIL
import cv2

import sneks
import gym

import torch
import torch.nn as nn

plt.style.use('default')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

info = dict(
    torch_version=torch.__version__,
    torch_device=DEVICE,
    gym_version=gym.__version__,
)


def use_virtual_display():
    # Necessary to display cartpole and other envs headlessly
    # https://stackoverflow.com/a/47968021
    from pyvirtualdisplay.smartdisplay import SmartDisplay as Display

    display = Display(visible=0, size=(1400, 900))
    display.start()

    return os.environ['DISPLAY']


def dense(i, o, a=nn.Sigmoid):
    l = nn.Linear(i, o)
    return nn.Sequential(l, a())


def lam(func):
    class Lambda(nn.Module):
        def forward(self, *args):
            return func(*args)

    return Lambda()


def model_persistor(model, path):
    def persist():
        torch.save(model.state_dict(), path)

    def load_if_exists():
        if os.path.isfile(path):
            model.load_state_dict(torch.load(path))

    return persist, load_if_exists


class DQNAgent(nn.Module):
    def __init__(self, obs_size, num_actions):
        super(DQNAgent, self).__init__()

        hidden_dim = 128

        def make_dqn():
            return nn.Sequential(
                nn.Flatten(),
                dense(obs_size, hidden_dim, nn.ReLU),
                dense(hidden_dim, hidden_dim, nn.ReLU),
                nn.Linear(hidden_dim, num_actions),
            ).to(DEVICE)

        self.eval_net = make_dqn()
        self.target_net = make_dqn().eval()

    def forward(self, obs):
        obs = torch.FloatTensor([obs]).to(DEVICE)
        q_vals = self.target_net(obs)[0]
        return q_vals.argmax().cpu().detach().numpy()

    def replace_target(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    # https://github.com/cyoon1729/deep-Q-networks/blob/master/vanillaDQN/dqn.py#L51
    def loss(self, i, batch):
        discount = 0.95
        criterion = nn.MSELoss()

        if i % 50:
            self.replace_target()

        obs, actions, rewards, next_obs, done = batch
        obs = torch.FloatTensor(obs).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_obs = torch.FloatTensor(next_obs).to(DEVICE)
        done = torch.BoolTensor(done).to(DEVICE)

        q_vals = self.eval_net(obs).gather(
            dim=1,
            index=actions.unsqueeze(0).T,
        ).squeeze()

        next_q_vals, _ = self.target_net(next_obs).max(dim=1)
        target_q_vals = next_q_vals * discount + rewards
        target_q_vals[done] = rewards[done]

        return criterion(q_vals, target_q_vals)


def get_experience_generator(
    env,
    model,
    bs,
    randomness=1,
    randomness_min=0.01,
    randomness_decay=0.9993,
):
    max_rollout_steps = 500
    buffer_size = 100_000

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
                action = model(obs)
            else:
                action = env.action_space.sample()

            next_obs, reward, done, _info = env.step(action)
            experience_pool.append((obs, action, reward, next_obs, done))
            obs = next_obs
            episode_rewards[-1] += reward

            if len(experience_pool) >= bs:
                batch = random.sample(experience_pool, bs)
                yield [np.array(t)
                       for t in zip(*batch)], episode_rewards, randomness_list


def dqn_optimize(env, model, its, next_batch, lr):
    tr = trange(its, bar_format="{bar}{l_bar}{r_bar}")
    fig, ((ax_loss, ax_reward, ax_eps)) = plt.subplots(1, 3, figsize=(12, 3))
    losses = []
    loss_avgs = []
    it_ids = []

    optimizer = torch.optim.Adam(params=model.eval_net.parameters(), lr=lr)

    for i in tr:
        it_ids.append(i)
        batch, episode_rewards, randomness_list = next(next_batch)

        optimizer.zero_grad()
        loss = model.loss(i, batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        loss_avgs.append(np.mean(losses[-20:]))

        description = [
            'it %i/%i' % (i + 1, its),
            'loss: %.6f' % loss.item(),
            'reward moving avg: %.2f' % np.mean(episode_rewards[-50:-1]),
            ' #### ',
        ]

        if i % 200 == 0:
            ax_loss.clear()
            ax_loss.plot(it_ids[-500:], losses[-500:], linewidth=1)
            ax_loss.plot(it_ids[-500:], loss_avgs[-500:], linewidth=2)

            ax_reward.clear()
            ax_reward.plot(episode_rewards[:-1], linewidth=2)

            ax_eps.clear()
            ax_eps.plot(randomness_list, linewidth=2)

            fig.canvas.draw()

        tr.set_description(' | '.join(description))


def play_env(env, agent, duration):
    time_step = 0

    while True:
        obs = env.reset()
        done = False

        while not done:
            time_step += 1
            action = agent(obs)
            obs, _reward, done, _info = env.step(action)
            yield env.render('rgb_array')

            if time_step >= duration: return


def i_python_display_frames(frames_generator, fps=100):
    # https://github.com/NicksonYap/Jupyter-Webcam/blob/master/Realtime_video_ipython_py3.ipynb
    def show_array(colorMatrix, prev_display_id=None, fmt='jpeg'):
        f = io.BytesIO()
        PIL.Image.fromarray(colorMatrix).save(f, fmt)
        obj = IPython.display.Image(data=f.getvalue())
        IPython.display.display(obj)

    def clear_display():
        IPython.display.clear_output(wait=True)

    try:
        for frame in frames_generator:
            time.sleep(1 / fps)
            frame = cv2.resize(frame, (256, 265))
            clear_display()
            show_array(frame)

    except KeyboardInterrupt as _e:
        clear_display()
        show_array(frame)  # show last frame
