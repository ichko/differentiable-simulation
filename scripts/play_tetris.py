from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
import time

import numpy as np
import matplotlib.pyplot as plt

import cv2
import keyboard

win_name = 'Tetris'

env = gym_tetris.make('TetrisA-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True


def get_keyboard_action():
    while True:
        # x = '0123456789'
        # x = [i for i in x if keyboard.is_pressed(i)]
        # x = [*x, '0']
        # return int(x[0])

        if keyboard.is_pressed('a'):
            return 1
        if keyboard.is_pressed('d'):
            return 2
        return 0


cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 500, 500)

for step in range(5000):
    time.sleep(1 / 60)
    if done:
        state = env.reset()
    action = env.action_space.sample()

    state, reward, done, info = env.step(action)
    for _ in range(3):
        state, reward, done, info = env.step(5)

    print(step)
    frame = env.render('rgb_array')
    frame = cv2.resize(frame, (256, 256))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow(win_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


env.close()
