import datetime
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as kl

import utils.tf_helpers as tfh


def de_conv(f, ks, s, a):
    return kl.TimeDistributed(
        kl.Conv2DTranspose(
            filters=f,
            kernel_size=(ks, ks),
            strides=(s, s),
            activation=a,
        ))


def mk_recurrence(size):
    return tf.keras.Sequential(
        [
            tfh.drnn(type='gru', size=size, skip=1, name='gru1'),
            kl.BatchNormalization(),
            tfh.drnn(type='gru', size=size, skip=4, name='gru2'),
            kl.BatchNormalization(),
            tfh.drnn(type='gru', size=size, skip=1, name='gru3'),
            kl.BatchNormalization(),
        ],
        name='memory',
    )


def mk_renderer():
    # There is a memory leak issue with using TimeDistributed
    # https://github.com/tensorflow/tensorflow/issues/33178
    start_size = 16
    return tf.keras.Sequential(
        [
            kl.Dense(start_size * start_size),
            kl.Reshape((-1, start_size, start_size, 1)),
            de_conv(f=128, ks=2, s=2, a='relu'),  # 32
            de_conv(f=64, ks=2, s=2, a='relu'),  # 64
            de_conv(f=16, ks=2, s=2, a='relu'),  # 128
            de_conv(f=3, ks=1, s=1, a='sigmoid'),
        ],
        name='renderer',
    )


def mk_reward():
    return tf.keras.Sequential(
        [
            kl.Dense(8, activation='relu', name='reward_dense1'),
            kl.Dense(1, activation='softmax', name='reward_dense2')
        ],
        name='reward',
    )


def mk_tb_callback():
    model_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = 'logs/fit/probabilistic-drnn/' + model_id

    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
    )


class DRNN:
    def __init__(self, internal_size, lr, weight_decay):
        self.tb_callback = mk_tb_callback()

        action = kl.Input(shape=(None, 3), name='action')
        self.memory = mk_recurrence(internal_size)
        self.renderer = mk_renderer()
        self.reward = mk_reward()

        latent_memory = self.memory(action)
        observation = self.renderer(latent_memory)
        reward = self.reward(latent_memory)

        self.net = tf.keras.Model([action], [observation, reward])
        self.net.compile(
            loss='binary_crossentropy',
            optimizer=tfa.optimizers.AdamW(
                learning_rate=lr,
                weight_decay=weight_decay,
            ),
        )
