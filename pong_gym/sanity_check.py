from gym.envs.classic_control import rendering
import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import random


def repeat_upsample(rgb_array, s=1):
    return np.repeat(np.repeat(rgb_array, s, axis=0), s, axis=1)


viewer = rendering.SimpleImageViewer()
env = gym.make('Pong-v0')
env.reset()

while range(1000):
    time.sleep(1 / 30)
    action = random.choices([1, 2, 3])[0]  # env.action_space.sample()
    observation, reward, done, info = env.step(action)
    upscaled = repeat_upsample(observation, 3)
    viewer.imshow(upscaled)
    # print(upscaled[:, :, 0].shape)
    # plt.imshow(upscaled)
    # plt.show()
