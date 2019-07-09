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

    state = random_state(W, H)
    renderer = Renderer(500, 500, 'Learned Conway')

    cell_size = 10
    while renderer.is_running:
        renderer.update()

        for x in range(W):
            for y in range(H):
                if state[x, y]:
                    pos_x = x - W / 2
                    pos_y = y - H / 2

                    renderer.rect(
                        pos_x * cell_size,
                        pos_y * cell_size,
                        cell_size,
                        cell_size,
                        fill='#fff'
                    )

        state = next_learned_frame(state)
