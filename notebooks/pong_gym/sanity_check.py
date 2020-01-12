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
env = gym.make('CarRacing-v0')
env.reset()

while range(1000):
    time.sleep(1 / 30)
    action = env.action_space.sample()
    _observation, reward, done, info = env.step(action)

    if done:
        env.reset()

    rgb = env.render('rgb_array')
    scale = 0.4
    frame = cv2.resize(
        rgb,
        (int(rgb.shape[0] * scale), int(rgb.shape[1] * scale)),
        interpolation=cv2.INTER_LINEAR,
    )

    viewer.imshow(frame)
    plt.imshow(frame)
    plt.show()

    print(frame.shape)

    # print(upscaled[:, :, 0].shape)
    # plt.imshow(upscaled)
    # plt.show()
