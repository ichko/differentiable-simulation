import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import utils.tf_helpers as tfh


def make_initializer(size):
    project_state = tf.keras.layers.Dense(
        size,
        name='project_activation',
        activation='relu',
    )

    to_cartesian = tf.keras.layers.Lambda(
        lambda x: tf.stack([tf.sin(x), tf.cos(x)], axis=1)[:, :, 0],
        name='polar_to_cartesian',
    )

    return lambda x: project_state(to_cartesian(x))


def make_memory(size, stateful):
    gru1 = tfh.drnn('gru', size, 1, stateful, 'gru1')
    gru2 = tfh.drnn('gru', size, 4, stateful, 'gru2')
    gru3 = tfh.drnn('gru', size, 16, stateful, 'gru3')
    gru4 = tfh.drnn('gru', size, 32, stateful, 'gru4')

    return lambda x, s=None: gru4(gru3(gru2(gru1(x, s))))


def make_render(W, H):
    renderer = tf.keras.layers.Dense(
        W * H,
        activation='sigmoid',
        name='frame_vector',
    )

    reshaper = tf.keras.layers.Reshape(
        (-1, W, H),
        name='frame_matrix',
    )

    return lambda x: reshaper(renderer(x))


class DRNN:
    def __init__(self, internal_size, W, H, lr, weight_decay, stateful=False):
        bs = 1 if stateful else None
        self.W = W
        self.H = H
        self.internal_size = internal_size
        self.stateful = stateful
        self.first_time = False

        model_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = 'logs/fit/drnn/' + model_id

        self.tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
        )

        game_init = tf.keras.layers.Input(
            shape=(1),
            name='game_init',
            batch_size=bs,
        )

        user_input = tf.keras.layers.Input(
            shape=(None, 2),
            name='user_input',
            batch_size=bs,
        )

        self.project_game_init = make_initializer(internal_size)
        self.rollout_memory = make_memory(internal_size, stateful)
        self.renderer = make_render(W, H)

        projected_init = self.project_game_init(game_init)
        memory = self.rollout_memory(user_input, projected_init)
        frame = self.renderer(memory)

        self.net = tf.keras.Model([game_init, user_input], frame)

        self.optimizer = tfa.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
        )

        self.net.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['mse'],
        )
