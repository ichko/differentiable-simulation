import time

import models

import numpy as np
import matplotlib.pyplot as plt
import gym
import cv2

win_name = 'win'


def render(env):
    frame = env.render('rgb_array')
    frame = cv2.resize(frame, (32, 32))
    return frame


def play():

    env = gym.make('CubeCrash-v0')
    env.reset()

    first = render(env)
    action = env.action_space.sample()
    env.step(action)
    second = render(env)

    preconditions = np.concatenate([first, second], axis=-1)
    preconditions = np.transpose(preconditions, (2, 0, 1))

    model = models.ForwardGym(
        num_actions=3,
        precondition=preconditions,
        action_output_channels=32,
        precondition_channels=2 * 3,
        precondition_out_channels=128,
    )
    model.make_persisted('.models/CubeCrash-v0.pkl')
    model.preload_weights()
    model.eval()
    model.reset()

    done = False
    while not done:
        time.sleep(1 / 5)

        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        pred_frame, _, _, _ = model.step(action)
        pred_frame = np.transpose(pred_frame, (1, 2, 0))
        frame = render(env)

        diff = abs(pred_frame.astype(np.float32) - frame).astype(np.uint8)

        screen = np.concatenate(
            [frame, pred_frame, diff], axis=1)

        cv2.imshow(win_name, screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 900, 300)

    while True:
        play()
