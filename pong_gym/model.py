import numpy as np
import tensorflow as tf


class Memory:
    def __init__(self, size, stateful=False):
        self.input_sequence_projection = tf.keras.layers.Dense(
            size,
            activation='tanh',
            name='input_sequence'
        )
        self.next_state = tf.keras.layers.LSTM(
            size,
            activation='tanh',
            return_sequences=True,
            stateful=stateful,
            name='next_state'
        )

    def __call__(self, input_sequence):
        x = self.input_sequence_projection(input_sequence)
        x = self.next_state(x)
        return x


class Vision:
    def __init__(self, output_shape):
        output_size = np.prod(output_shape)
        print(output_size)
        self.projection = tf.keras.layers.Dense(
            output_size,
            activation='sigmoid',
            name='vision_projection'
        )
        self.reshape = tf.keras.layers.Reshape(
            (-1, *output_shape),
            name='vision_reshape'
        )

    def __call__(self, latent_memory):
        x = self.projection(latent_memory)
        x = self.reshape(x)
        return x


class EnvModel:
    def __init__(self, memory_size, output_shape, stateful=False):
        bs = 1 if stateful else None

        # One hot of tree actions - top, bottom, stay
        user_input_size = 3
        user_input = tf.keras.layers.Input(
            shape=(None, user_input_size),
            batch_size=bs,
            name='user_input'
        )
        self.reward = tf.keras.layers.Dense(
            1, activation='tanh', name='reward'
        )
        self.done = tf.keras.layers.Dense(1, activation='sigmoid', name='done')

        self.memory = Memory(size=memory_size, stateful=stateful)
        self.vision = Vision(output_shape=output_shape)

        m = self.memory(user_input)
        v = self.vision(m)
        r = self.reward(m)
        d = self.done(m)

        self.net = tf.keras.Model([user_input], [v, r, d])

        self.net.compile(
            loss='mse',
            optimizer='adam',
            metrics=['mse', 'accuracy']
        )
