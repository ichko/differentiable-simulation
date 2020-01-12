import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import datetime
import utils.tf_helpers as tf_helpers

tf_print = tf_helpers.tf_print()


def make_memory(size, stateful):
    gru1 = tf_helpers.drnn_layer('gru', size, 1, stateful, 'gru1')
    gru2 = tf_helpers.drnn_layer('gru', size, 4, stateful, 'gru2')
    gru3 = tf_helpers.drnn_layer('gru', size, 1, stateful, 'gru3')

    return lambda x, s=None: gru3(gru2(gru1(x, s)))


def make_render(W, H):
    # There is a memory leak issue with using TimeDistributed
    # https://github.com/tensorflow/tensorflow/issues/33178

    d = tf.keras.layers.Dense(13 * 13)
    dr = tf.keras.layers.Reshape((-1, 13, 13, 1))

    l1 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2DTranspose(
            8,
            (2, 2),
            strides=(2, 2),
            activation='relu',
        ))

    l2 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2DTranspose(
            3,
            (2, 2),
            strides=(2, 2),
            activation='sigmoid',
        ))

    return lambda x: l2(l1(dr(d(x))))


def make_reward_projector():
    dense1 = tf.keras.layers.Dense(
        8,
        name='reward_dense1',
        activation='relu',
    )

    dense2 = tf.keras.layers.Dense(
        1,
        name='reward_dense2',
        activation='sigmoid',
    )

    return lambda x: dense2(dense1(x))


class DRNN:
    def __init__(self, internal_size, W, H, lr, weight_decay, stateful=False):
        bs = 1 if stateful else None
        self.W = W
        self.H = H
        self.internal_size = internal_size
        self.stateful = stateful
        self.first_time = False

        model_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = 'logs/fit/probabilistic-drnn/' + model_id

        self.tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
        )

        action = tf.keras.layers.Input(
            shape=(None, 3),
            name='action',
            batch_size=bs,
        )

        self.rollout_memory = make_memory(internal_size, stateful)
        self.renderer = make_render(W, H)
        self.reward_projection = make_reward_projector()

        memory = self.rollout_memory(action)
        frame = self.renderer(memory)
        reward = self.reward_projection(memory)

        self.net = tf.keras.Model([action], [frame, reward])

        self.optimizer = tfa.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
        )

        self.net.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            # metrics=['mse'],
        )
