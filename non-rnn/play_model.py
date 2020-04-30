import time

import numpy as np
import matplotlib.pyplot as plt
import gym
import cv2

from argparse import Namespace

import models
import utils

win_name = 'win'


def render_screen(frames_generator):
    for frame in frames_generator:
        cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 900, 300)

    env = gym.make('CubeCrash-v0')

    model = models.ForwardGym(
        num_actions=3,
        action_output_channels=32,
        precondition_channels=2 * 3,
        precondition_out_channels=128,
    )
    model.make_persisted('.models/CubeCrash-v0.pkl')
    model.preload_weights()

    # utils.produce_video('./.videos/test.webm', model, env)

    while True:
        rollout = utils.play_model(env, model)
        render_screen(rollout)
