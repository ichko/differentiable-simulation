import time

import numpy as np
from gym.envs.classic_control import rendering
import tensorflow as tf

from data import env_sequences_generator
from model import EnvModel


def repeat_upsample(rgb_array, s=1):
    return np.repeat(np.repeat(rgb_array, s, axis=0), s, axis=1)


def make_dataset(bs, seq_len):
    return tf.data.Dataset.from_generator(
        generator=lambda: env_sequences_generator(seq_len),
        output_types=((tf.float32,), (tf.float32, tf.float32, tf.bool)),
    ).batch(bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


if __name__ == '__main__':
    model = EnvModel(memory_size=32, output_shape=(210, 160, 3))
    dataset = make_dataset(bs=32, seq_len=128)

    print(model.net.summary())

    model.net.fit_generator(
        generator=dataset,
        validation_data=dataset,
        validation_steps=2,
        steps_per_epoch=100,
        epochs=250
    )
