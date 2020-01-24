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


def mk_sampler(bs, seq_len, internal_size):
    def sampler(x):
        normal = tf.random.normal((bs, seq_len, internal_size // 2))
        mean, stddev = tf.split(
            x,
            [internal_size // 2, internal_size // 2],
            axis=2,
        )
        return normal * stddev + mean

    return tf.keras.layers.Lambda(sampler, name='sampler')


def mk_initializer(input_shape, output_size):
    return tf.keras.Sequential(
        [
            kl.Input(input_shape),
            kl.Flatten(),
            kl.Dense(16, activation='relu', name='init1'),
            kl.BatchNormalization(),
            kl.Dense(32, activation='relu', name='init2'),
            kl.BatchNormalization(),
            kl.Dense(output_size, activation='tanh', name='init3'),
            kl.BatchNormalization(),
        ],
        name='initializer',
    )


def mk_recurrence(input_shape, internal_size):
    i = kl.Input(input_shape, name='actions_input')
    s = kl.Input(internal_size, name='initial_state')

    x = tfh.drnn(type='gru', size=internal_size, skip=1, name='gru1')(i, s)
    x = kl.BatchNormalization()(x)
    x = tfh.drnn(type='gru', size=internal_size, skip=4, name='gru2')(x)
    x = kl.BatchNormalization()(x)
    x = tfh.drnn(type='gru', size=internal_size, skip=1, name='gru3')(x)
    x = kl.BatchNormalization()(x)

    return tf.keras.Model([i, s], x, name='memory')


def mk_renderer(input_size):
    # There is a memory leak issue with TimeDistributed
    # https://github.com/tensorflow/tensorflow/issues/33178
    start_size = 16
    return tf.keras.Sequential(
        [
            kl.Input((None, input_size)),
            kl.Dense(start_size * start_size),
            kl.BatchNormalization(),
            kl.Reshape((-1, start_size, start_size, 1)),
            de_conv(f=128, ks=2, s=2, a='relu'),  # 32
            kl.BatchNormalization(),
            de_conv(f=64, ks=2, s=2, a='relu'),  # 64
            kl.BatchNormalization(),
            de_conv(f=16, ks=2, s=2, a='relu'),  # 128
            kl.BatchNormalization(),
            de_conv(f=3, ks=1, s=1, a='sigmoid'),
        ],
        name='renderer',
    )


def mk_reward(input_size):
    return tf.keras.Sequential(
        [
            kl.Input((None, input_size)),
            kl.Dense(8, activation='relu', name='reward_dense1'),
            kl.BatchNormalization(),
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
    def __init__(self, seq_len, bs, internal_size, lr, weight_decay):
        self.tb_callback = mk_tb_callback()

        action_shape = (None, 3)
        init_shape = (12, 2)
        condition = kl.Input(shape=init_shape, name='condition')
        action = kl.Input(shape=action_shape, name='action')

        self.init = mk_initializer(init_shape, internal_size)
        self.memory = mk_recurrence(action_shape, internal_size)
        self.sampler = mk_sampler(bs, seq_len, internal_size)
        self.renderer = mk_renderer(internal_size // 2)
        self.reward = mk_reward(internal_size // 2)

        init = self.init(condition)
        latent_memory = self.memory([action, init])
        print(latent_memory.shape)
        latent_memory = self.sampler(latent_memory)

        observation = self.renderer(latent_memory)
        reward = self.reward(latent_memory)

        self.net = tf.keras.Model([condition, action], [observation, reward])
        self.net.compile(
            loss='binary_crossentropy',
            optimizer=tfa.optimizers.AdamW(
                learning_rate=lr,
                weight_decay=weight_decay,
            ),
        )
