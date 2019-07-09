from game_of_life import next_frame, random_state
from renderer import Renderer

import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    W, H = 30, 30
    MODEL = tf.keras.models.load_model(f'conway_{W}x{H}_small_error.h5')

    def next_learned_frame(state):
        state = state.reshape(1, W, H, 1)
        pred = MODEL.predict(state)
        return np.squeeze(pred) > 0.5

    state_learned = random_state(W, H)
    state_actual = state_learned.copy()
    renderer = Renderer(500, 500, 'Learned Conway')

    cell_size = 10
    while renderer.is_running:
        renderer.update()

        for x in range(W):
            for y in range(H):
                pos_x = x - W / 2
                pos_y = y - H / 2

                color = '#222'

                if state_actual[x, y] and state_learned[x, y]:
                    color = '#fff'
                elif state_learned[x, y]:
                    color = '#f0f'
                elif state_actual[x, y]:
                    color = '#0ff'

                renderer.rect(
                    pos_x * cell_size,
                    pos_y * cell_size,
                    cell_size,
                    cell_size,
                    fill=color
                )

        state_learned = next_learned_frame(state_learned)
        state_actual = next_frame(state_actual)
