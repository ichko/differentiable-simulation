import IPython

import os
import io
import time
import random
from collections import deque, defaultdict

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

    def optimizer(self, next_batch, lr):
        optim = torch.optim.Adam(params=self.eval_net.parameters(), lr=lr)

        for i, batch in enumerate(next_batch):
            optim.zero_grad()
            loss = self.loss(i, batch)
            loss.backward()
            optim.step()

            yield dict(loss=loss.item())


class ExperienceGenerator:
    def __call__(
        self,
        env,
        model,
        bs,
        randomness=1,
        randomness_min=0.01,
        randomness_decay=0.9993,
        max_rollout_steps=500,
        buffer_size=100_000,
    ):
        self.episode_rewards = []
        self.randomness_list = []
        experience_pool = deque(maxlen=buffer_size)

        while True:
            obs = env.reset()
            done = False
            step = 0
            self.episode_rewards.append(0)

            while not done and step < max_rollout_steps:
                randomness = max(randomness_min, randomness * randomness_decay)
                self.randomness_list.append(randomness)
                step += 1

                use_model = random.uniform(0, 1) > randomness
                if use_model:
                    action = model(obs)
                else:
                    action = env.action_space.sample()

                next_obs, reward, done, _info = env.step(action)
                experience_pool.append((obs, action, reward, next_obs, done))
                obs = next_obs
                self.episode_rewards[-1] += reward

                if len(experience_pool) >= bs:
                    batch = random.sample(experience_pool, bs)
                    yield [np.array(t) for t in zip(*batch)]


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


class Plotter:
    def __init__(self, rows=1):
        self.rows = rows
        self.history = defaultdict(lambda: [])
        self.history_avg = defaultdict(lambda: [])

    def log(self, **info):
        for name, value in sorted(info.items()):
            if type(value) is list:
                self.history[name] = value
            else:
                self.history[name].append(value)
                vals_so_far = self.history[name]

                avg_size = 20
                self.history_avg[name].append(np.mean(vals_so_far[-avg_size:]))

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

        window_plot_size = 500

        for ax, (name, _value) in zip(self.axs, sorted(self.history.items())):
            ax.clear()
            ax.set_title(name)

            ax.plot(self.history[name][-window_plot_size:], linewidth=2)
            if name in self.history_avg:
                ax.plot(self.history_avg[name][-window_plot_size:],
                        linewidth=2)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class RNNWorldRepresentations(nn.Module):
    def __init__(self, obs_shape, actions_shape, num_rollouts):
        super(RNNWorldRepresentations, self).__init__()

        obs_size = np.prod(obs_shape)
        self.rnn_num_layers = 2
        rnn_inp_size = 128
        rnn_hidden_size = 64

        self.precondition = nn.Embedding(
            num_embeddings=num_rollouts,
            embedding_dim=rnn_hidden_size * 2,
        )

        self.action_encoder = nn.Sequential(
            nn.Flatten(),
            dense(np.prod(actions_shape), rnn_inp_size),
        )

        self.time_transition = nn.GRU(
            rnn_inp_size,
            rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True,
        )

        self.obs_decoder = nn.Sequential(
            dense(rnn_hidden_size, 512, nn.ReLU),
            dense(512, 512, nn.ReLU),
            dense(512, obs_size, nn.Sigmoid),
            lam(lambda x: x.reshape(*obs_shape)),
        )

    def optimizer(self, next_batch, lr):
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()

        for (idx, actions), obs in next_batch:
            optim.zero_grad()

            preconditions = self.precondition(idx)
            preconditions = torch.stack(
                preconditions.chunk(self.rnn_num_layers, dim=1),
                dim=0,
            )

            encoded_actions = self.action_encoder(actions)
            memory = self.time_transition(encoded_actions, preconditions)
            pred_obs = self.obs_decoder(memory)

            loss = criterion(pred_obs, obs)
            loss.backward()
            optim.step()

            yield dict(loss=loss.item())


def rollout_generator():
    # TODO: Implement
    pass
