import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import utils.rnn_model
import utils.pong
import utils.tensorflow
from utils.renderer import Renderer
from utils.renderer import get_pressed_key

import tensorflow as tf
from matplotlib import cm
import numpy as np


def cmap(x):
    return cm.bwr(1 - x)


def main():
    model = utils.rnn_model.Model(32, 40, 40, 0.005, 0.0001)
    utils.tensorflow.model_persistor(
        model, checkpoint_dir='./notebooks/.checkpoints/')
    model = model.copy_in_stateful_model()

    Renderer.init_window(1000, 500)

    while True:
        angle = np.random.uniform(0, 2 * np.pi)
        direction = [np.sin(angle), np.cos(angle)]

        simulation = utils.pong.PONGSimulation(W=40, H=40, direction=angle)
        model.init(direction)

        while True:
            key = get_pressed_key()

            movement_left = 1 if key == 'w' else 0
            movement_left = -1 if key == 's' else movement_left

            movement_right = 1 if key == 'up' else 0
            movement_right = -1 if key == 'down' else movement_right

            controls = [movement_left, movement_right]

            frame, _ = simulation.tick(controls)
            pred_frame = model.tick(controls)

            rgb_frame = cmap(frame)
            rgb_pred_frame = cmap(pred_frame)

            print('LEFT: [%s]  --  RIGHT: [%s]' %
                  tuple(' UP ' if i > 0 else 'DOWN' if i < 0 else ' ## '
                        for i in controls))

            split_screen = np.concatenate((rgb_frame, rgb_pred_frame), axis=1)
            Renderer.show_frame(split_screen)

            if key == 'q': return
            if key == 'r': break


if __name__ == '__main__':
    main()