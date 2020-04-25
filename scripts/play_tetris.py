from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
import time

import numpy as np
import matplotlib.pyplot as plt

env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
for step in range(5000):
    time.sleep(1 / 60)
    if done:
        state = env.reset()
    action = env.action_space.sample()

    state, reward, done, info = env.step(action)
    # for _ in range(3):
    #     state, reward, done, info = env.step(5)

    print(step)
    env.render()

env.close()