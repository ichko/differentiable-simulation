import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import datetime


class Model:
    def __init__(self, internal_size, W, H, lr, weight_decay, stateful=False):
        bs = 1 if stateful else None
        self.W = W
        self.H = H
        self.internal_size = internal_size
        self.stateful = stateful
        self.first_time = False

        model_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = 'logs/fit/' + model_id

        self.tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)

        game_init = tf.keras.layers.Input(shape=(2),
                                          name='game_init',
                                          batch_size=bs)
        user_input = tf.keras.layers.Input(shape=(None, 2),
                                           name='user_input',
                                           batch_size=bs)

        initial_state = self.project_game_init(game_init)
        rollout_memory = self.rollout(initial_state, user_input)
        frame = self.render(rollout_memory)

        self.net = tf.keras.Model([game_init, user_input], frame)

        initial_learning_rate = lr

        self.optimizer = tfa.optimizers.AdamW(
            learning_rate=initial_learning_rate, weight_decay=weight_decay)

        self.net.compile(loss='binary_crossentropy',
                         optimizer=self.optimizer,
                         metrics=['mse'])

    def project_game_init(self, init_x):
        if not getattr(self, 'project_state', False):
            self.project_state = tf.keras.layers.Dense(
                self.internal_size,
                name='project_activation',
                activation='relu')

        return self.project_state(init_x)

    def rollout(self, initial_state, user_input):
        if not getattr(self, 'gru', False):
            self.gru = tf.keras.layers.GRU(self.internal_size,
                                           return_sequences=True,
                                           stateful=self.stateful,
                                           name='gru',
                                           activation='tanh')

            self.bn = tf.keras.layers.BatchNormalization()

            self.gru2 = tf.keras.layers.GRU(self.internal_size,
                                            return_sequences=True,
                                            stateful=self.stateful,
                                            name='gru2',
                                            activation='tanh')

            self.bn2 = tf.keras.layers.BatchNormalization()

        x = self.gru(user_input, initial_state=initial_state)
        x = self.bn(x)
        x = self.gru2(x)
        x = self.bn2(x)

        return x

    def render(self, memory):
        if not getattr(self, 'renderer', False):
            self.renderer = tf.keras.layers.Dense(self.W * self.H,
                                                  activation='sigmoid',
                                                  name='frame_vector')
            self.reshaper = tf.keras.layers.Reshape((-1, self.W, self.H),
                                                    name='frame_matrix')

        return self.reshaper(self.renderer(memory))

    def copy_in_stateful_model(self):
        stateful = Model(self.internal_size, self.W, self.H, 0.1, 0.1, True)

        for nb, layer in enumerate(self.net.layers):
            stateful.net.layers[nb].set_weights(layer.get_weights())

        return stateful

    def init(self, direction):
        direction = np.expand_dims(np.array(direction), axis=0)
        self.init_dir = tf.convert_to_tensor(direction, dtype=tf.float32)
        self.first_time = True

    def tick(self, user_input):
        user_input = np.array(user_input).reshape((1, 1, -1))
        user_input = tf.convert_to_tensor(user_input, dtype=tf.float32)

        initial_state = self.project_game_init(self.init_dir)
        rollout_memory = self.rollout(
            initial_state if self.first_time else None, user_input)
        frame = self.render(rollout_memory)

        self.first_time = False

        return frame[0][0]