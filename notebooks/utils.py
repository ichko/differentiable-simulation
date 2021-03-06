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


def dense(i, o, a=nn.Sigmoid):
    l = nn.Linear(i, o)
    return nn.Sequential(l, a())


def lam(func):
    class Lambda(nn.Module):
        def forward(self, *args):
            return func(*args)

    return Lambda()


def use_virtual_display():
    # Necessary to display cartpole and other envs headlessly
    # https://stackoverflow.com/a/47968021
    from pyvirtualdisplay.smartdisplay import SmartDisplay as Display

    display = Display(visible=0, size=(1400, 900))
    display.start()

    return os.environ['DISPLAY']


class PersistedModel:
    def make_persisted(self, path):
        self.path = path

    def persist(self):
        torch.save(self.state_dict(), self.path)

    def load_persisted_if_exists(self):
        if os.path.isfile(self.path):
            self.load_state_dict(torch.load(self.path))

    def can_be_preloaded(self):
        return os.path.isfile(self.path)


class DQNAgent(nn.Module, PersistedModel):
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
    def loss(self, i, obs, actions, rewards, next_obs, done):
        discount = 0.95
        criterion = nn.MSELoss()

        if i % 50:
            self.replace_target()

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
            loss = self.loss(i, **batch)
            loss.backward()
            optim.step()

            yield dict(loss=loss.item())


class ExperienceGenerator:
    def __init__(self, *args, **kwargs):
        self.generator = self.make_generator(*args, **kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)

    def make_generator(
        self,
        env,
        agent,
        bs,
        randomness=1,
        randomness_min=0.01,
        randomness_decay=0.9993,
        max_rollout_steps=500,
        buffer_size=100_000,
    ):
        self.episode_rewards = []
        self.randomness_list = []
        self.buffer = deque(maxlen=buffer_size)
        self.episodes_len = []

        while True:
            obs = env.reset()
            done = False
            step = 0
            self.episodes_len.append(0)
            self.episode_rewards.append(0)

            while not done and step < max_rollout_steps:
                randomness = max(randomness_min, randomness * randomness_decay)
                self.randomness_list.append(randomness)
                step += 1

                use_model = random.uniform(0, 1) > randomness
                if use_model: action = agent(obs)
                else: action = env.action_space.sample()

                next_obs, reward, done, _info = env.step(action)
                self.buffer.append((obs, action, reward, next_obs, done))
                obs = next_obs
                self.episode_rewards[-1] += reward
                self.episodes_len[-1] += 1

                if len(self.buffer) >= bs:
                    batch = random.sample(self.buffer, bs)
                    b_obs, b_actions, b_rewards, b_next_obs, b_done = [
                        np.array(t) for t in zip(*batch)
                    ]

                    yield dict(
                        obs=b_obs / 255,
                        actions=b_actions,
                        rewards=b_rewards,
                        next_obs=b_next_obs / 255,
                        done=b_done,
                    )


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


def display_video(path):
    from base64 import b64encode
    from IPython.display import HTML

    video = open(path, 'rb')
    data_url = "data:video/webm;base64," + b64encode(video.read()).decode()
    video.close()
    return HTML('<video autoplay loop><source src="%s"></video>' % data_url)


def frames_to_video(name, frames, fps=30, frame_size=None):
    frames_iterator = iter(frames)
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    frame = next(frames_iterator)
    H, W = frame.shape[:2]
    frame_size = (W, H) if frame_size is None else frame_size
    video = cv2.VideoWriter(name, fourcc, fps, frame_size)

    for frame in frames_iterator:
        frame = cv2.resize(frame, frame_size)
        if frame.max() <= 1: frame *= 255
        frame = frame.astype(np.uint8)

        video.write(frame)

    video.release()


def i_python_display_frames(frames, fps=100, frame_size=None):
    # https://github.com/NicksonYap/Jupyter-Webcam/blob/master/Realtime_video_ipython_py3.ipynb
    def show_array(colorMatrix, prev_display_id=None, fmt='png'):
        f = io.BytesIO()
        PIL.Image.fromarray(colorMatrix).save(f, fmt)
        obj = IPython.display.Image(data=f.getvalue())
        IPython.display.display(obj)

    def clear_display():
        IPython.display.clear_output(wait=True)

    try:
        for frame in frames:
            time.sleep(1 / fps)

            # frame_min, frame_max = frame.min(), frame.max()
            # frame = (frame + frame_min) / (frame_max - frame_min) * 255
            if frame.max() <= 1:
                frame *= 255

            frame = frame.astype(np.uint8)
            if frame_size is None:
                frame_size = frame.shape[:2][::-1]  # (W, H)

            frame = cv2.resize(frame, frame_size)

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


class RNNWorldRepresentations(nn.Module, PersistedModel):
    def __init__(self, obs_shape, num_actions, seq_len, num_rollouts):
        super(RNNWorldRepresentations, self).__init__()

        obs_size = np.prod(obs_shape)
        self.rnn_num_layers = 2
        rnn_inp_size = 16
        rnn_hidden_size = 16

        self.precondition = nn.Embedding(
            num_embeddings=num_rollouts,
            embedding_dim=rnn_hidden_size * 2,
        ).to(DEVICE)

        self.action_encoder = dense(
            num_actions,
            rnn_inp_size,
            a=nn.Tanh,
        ).to(DEVICE)

        self.time_transition = nn.GRU(
            rnn_inp_size,
            rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True,
        ).to(DEVICE)

        self.obs_decoder = nn.Sequential(
            dense(rnn_hidden_size, 128, nn.ReLU),
            dense(128, 128, nn.ReLU),
            dense(128, obs_size, nn.Sigmoid),
            lam(lambda x: x.reshape(-1, seq_len, *obs_shape)),
        ).to(DEVICE)

    def forward(self, ids, actions):
        ids = torch.LongTensor(ids).to(DEVICE)
        actions = torch.FloatTensor(actions).to(DEVICE)

        preconditions = self.precondition(ids)
        preconditions = torch.stack(
            preconditions.chunk(self.rnn_num_layers, dim=1),
            dim=0,
        )

        encoded_actions = self.action_encoder(actions)
        memory, _ = self.time_transition(encoded_actions, preconditions)
        pred_obs = self.obs_decoder(memory)

        return pred_obs

    def optimizer(self, next_batch, lr):
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()

        for (ids, actions), obs in next_batch:
            obs = torch.FloatTensor(obs).to(DEVICE)

            optim.zero_grad()
            pred_obs = self(ids, actions)
            loss = criterion(pred_obs, obs)
            loss.backward()
            optim.step()

            yield dict(loss=loss.item())


def batch_info(batch):
    return {
        k: dict(shape=t.shape, max=t.max(), min=t.min())
        for k, t in batch.items()
    }


class RolloutGenerator:
    def __init__(self, *args, **kwargs):
        self.generator = self.make_generator(*args, **kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)

    def make_generator(
        self,
        env,
        agent,
        bs,
        max_seq_len,
        buffer_size,
        frame_size=None,
    ):
        ''' Yields batches of episodes - (ep_id, actions, observations, frames?) '''

        self.buffer = []
        self.episodes_len = []

        def get_batch():
            batch = random.sample(self.buffer, bs)
            ep_id, actions, obs, frames = [np.array(t) for t in zip(*batch)]
            return dict(ep_id=ep_id, actions=actions, obs=obs, frames=frames)

        for ep_id in range(buffer_size):
            obs = env.reset()
            done = False

            actions = np.zeros((max_seq_len, *env.action_space.shape))
            observations = np.zeros((max_seq_len, *obs.shape))
            self.episodes_len.append(0)

            frame = env.render('rgb_array')
            # Reverse to preserve in (W, H) form
            frame_size = frame.shape[:2][::-1] \
                if frame_size is None else frame_size

            # Assume RGB (3 channel) rendering
            frames = np.zeros((max_seq_len, *frame_size[::-1], 3))

            for i in range(max_seq_len):
                action = agent(obs)
                actions[i] = action
                observations[i] = obs
                frame = env.render('rgb_array')
                frames[i] = cv2.resize(frame, frame_size)

                obs, _reward, done, _info = env.step(action)
                self.episodes_len[-1] += 1

                if len(self.buffer) >= bs: yield get_batch()
                if done: break

            self.buffer.append([ep_id, actions, observations, frames])

        while True:
            yield get_batch()


def one_hot(data):
    flat_data = data.reshape(-1)
    one_hot = np.zeros((flat_data.size, data.max() + 1))
    rows = np.arange(flat_data.size)
    one_hot[rows, data.reshape(-1)] = 1

    return one_hot.astype(np.int32).reshape((*data.shape, -1))


def tile_frame_generators(generator_A, generator_B):
    for left_frame, right_frame in zip(generator_A, generator_B):
        H, _W = left_frame.shape[:2]
        white_line = np.full((H, 1, 3), 255.0)

        yield np.concatenate((left_frame, white_line, right_frame), axis=1)
