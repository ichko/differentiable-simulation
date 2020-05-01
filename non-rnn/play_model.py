import time

import numpy as np
import matplotlib.pyplot as plt
import gym
import cv2

from argparse import Namespace

from run_experiment import MODEL
import utils
import keyboard

win_name = 'win'


def get_keyboard_action():
    while True:
        # x = '0123456789'
        # x = [i for i in x if keyboard.is_pressed(i)]
        # x = [*x, '0']
        # return int(x[0])

        if keyboard.is_pressed('a'):
            return 1
        if keyboard.is_pressed('d'):
            return 2
        return 0


def render_screen(frames_generator):
    for frame in frames_generator:
        time.sleep(1 / 10)
        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 900, 300)

    env = gym.make('CubeCrash-v0')

    while True:
        rollout = utils.play_model(env, MODEL, agent=None)
        render_screen(rollout)
