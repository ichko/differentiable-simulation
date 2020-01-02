import tensorflow as tf
import tensorflow_addons as tfa
import datetime


class Model:
    def __init__(self, internal_size, W, H):
        self.W = W
        self.H = H
        self.internal_size = internal_size

        model_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = 'logs/fit/' + model_id

        self.tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1)

        game_init = tf.keras.layers.Input(shape=(2), name='game_init')
        user_input = tf.keras.layers.Input(shape=(None, 2), name='user_input')

        initial_state = self.project_game_init(game_init)
        rollout_memory = self.rollout(initial_state, user_input)
        frame = self.render(rollout_memory)

        self.net = tf.keras.Model([game_init, user_input], frame)

        initial_learning_rate = 0.005

        self.optimizer = tfa.optimizers.AdamW(
            learning_rate=initial_learning_rate, weight_decay=1e-4)

        self.net.compile(loss='binary_crossentropy',
                         optimizer=self.optimizer,
                         metrics=['mse'])

    def project_game_init(self, init_x):
        self.project_state = tf.keras.layers.Dense(self.internal_size,
                                                   name='project_activation',
                                                   activation='relu')

        return self.project_state(init_x)

    def rollout(self, initial_state, user_input):
        self.gru = tf.keras.layers.GRU(self.internal_size,
                                       return_sequences=True,
                                       name='gru',
                                       activation='tanh')

        self.gru2 = tf.keras.layers.GRU(self.internal_size,
                                        return_sequences=True,
                                        name='gru3',
                                        activation='tanh')
        x = self.gru(user_input, initial_state=initial_state)
        x = self.gru2(x)

        return x

    def render(self, memory):
        self.renderer = tf.keras.layers.Dense(self.W * self.H,
                                              activation='sigmoid',
                                              name='frame_matrix')
        self.reshaper = tf.keras.layers.Reshape((-1, self.W, self.H),
                                                name='frame_vector')

        return self.reshaper(self.renderer(memory))