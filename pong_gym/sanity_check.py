from gym.envs.classic_control import rendering
import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2


def repeat_upsample(rgb_array, s=1):
    return np.repeat(np.repeat(rgb_array, s, axis=0), s, axis=1)


viewer = rendering.SimpleImageViewer()
env = gym.make('PongDeterministic-v4')
env.reset()

while range(1000):
    time.sleep(1 / 30)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        env.reset()
    f = cv2.resize(observation, (80, 100), interpolation=cv2.INTER_LINEAR)
    upscaled = repeat_upsample(f, 5)
    viewer.imshow(upscaled)

    # print(upscaled[:, :, 0].shape)
    # plt.imshow(upscaled)
    # plt.show()
