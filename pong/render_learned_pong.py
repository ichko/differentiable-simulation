from renderer import Renderer
from math import copysign, sin

import tensorflow as tf
import numpy as np

tf.enable_eager_execution()


class Model:
    def __init__(self, net):
        _user_input, _dir_input, transformed_user_input, \
            activation_direction, hidden_direction, x1, x2, x3, \
            frames1, game_over1, frames2, game_over2 = net.layers

        self.transformed_user_input = transformed_user_input
        self.activation_direction = activation_direction
        self.hidden_direction = hidden_direction
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.frames1 = frames1
        self.game_over1 = game_over1
        self.frames2 = frames2
        self.game_over2 = game_over2

    def init(self, direction):
        direction = np.array(direction).reshape((1, 1))
        self.init_dir = tf.convert_to_tensor(direction, dtype=tf.float32)
        self.first_time = True

    def single_step_predict(self, user_input):
        user_input = np.array(user_input).reshape((1, 1, 2))
        user_input = tf.convert_to_tensor(user_input, dtype=tf.float32)
        ui = self.transformed_user_input(user_input)

        if self.first_time:
            self.first_time = False

            hd = self.hidden_direction(self.init_dir)
            ad = self.activation_direction(self.init_dir)
            x1 = self.x1(ui, initial_state=[ad, hd])
        else:
            x1 = self.x1(ui)

        x2 = self.x2(x1)
        x3 = self.x3(x2)
        f1 = self.frames1(x3)
        f2 = self.frames2(f1)
        g1 = self.game_over1(x3)
        g2 = self.game_over2(g1)

        return f2.numpy().reshape(50, 50), g2.numpy()

if __name__ == '__main__':
    keras_network = tf.keras.models.load_model('STATEFUL_LSTM3_PONG.hdf5')

    model = Model(keras_network)
    model.init(0.4)
    Renderer.init_window()

    f = 0
    while Renderer.can_render():
        f += 1

        left_dir = copysign(1, sin(f / 10))
        right_dir = copysign(1, sin(f / 12 + 1.2))

        print('LEFT: [%s]  --  RIGHT: [%s]' % (' UP ' if left_dir > 0 else 'DOWN',
                                               ' UP ' if right_dir > 0 else 'DOWN'))

        frame, _ = model.single_step_predict([left_dir, right_dir])
        empty_channel = np.zeros_like(frame)

        rgb_frame = np.stack((frame / 2, frame > 0.5, frame > 0.2), axis=-1)

        Renderer.show_frame(rgb_frame)
