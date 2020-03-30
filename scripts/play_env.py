import gym
import keyboard
import time

if __name__ == '__main__':
    env = gym.make('CubeCrash-v0')

    print(f'action space - {env.action_space}')

    def get_keyboard_action():
        while True:
            if keyboard.is_pressed('a'): return 1
            if keyboard.is_pressed('d'): return 2
            return 0

    while True:
        done = False
        env.reset()
        score = 0

        while not done:
            time.sleep(1 / 30)

            action = env.action_space.sample()
            action = get_keyboard_action()

            obs, reward, done, _info = env.step(action)
            score += reward
            env.render('human')

            print(reward)

        print(f'GAME OVER {score}')
        # keyboard.wait('esc')
