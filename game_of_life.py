import numpy as np


def random_state(width, height):
    return np.random.randint(2, size=(width, height))


def next_frame(state):
    new_state = state.copy()
    width, height = state.shape

    for x in range(width):
        for y in range(height):
            arround = population_arround(state, x, y)
            if state[y, x] and (arround < 2 or arround > 3):
                new_state[y, x] = 0
            elif arround == 3:
                new_state[y, x] = 1

    return new_state


def population_arround(state, x, y):
    width, height = state.shape

    def is_alive(x, y):
        # Torus world
        # x, y = x % width, y % height
        if x < 0 or x >= width:
            return 0
        if y < 0 or y >= height:
            return 0
        return state[y, x]

    return sum([
        is_alive(x + dx, y + dy)
        for dx in [-1, 0, 1]
        for dy in [-1, 0, 1]
        if dx != 0 or dy != 0
    ])


if __name__ == '__main__':
    state = random_state(10, 10)

    for _ in range(3):
        print(state)
        state = next_frame(state)
