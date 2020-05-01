import gym
import keyboard
import time

if __name__ == '__main__':
    # BattleZone-ram-v0 - 3D tank
    # Atlantis-ram-v0 - visually interesting (protect city)
    # Asteroids-ram-v0
    # Freeway-v0 - frogger like - looks promising

    env = gym.make('FlappyBird-v0')

    print(f'action space - {env.action_space}')

    def get_keyboard_action():
        while True:
            x = '0123456789'
            x = [i for i in x if keyboard.is_pressed(i)]
            x = [*x, '0']
            return int(x[0])

    while True:
        done = False
        env.reset()
        score = 0

        while not done:
            time.sleep(1 / 30)

            action = env.action_space.sample()
            # action = 2
            action = get_keyboard_action()

            obs, reward, done, _info = env.step(action)
            score += reward
            env.render('human')

            # print(reward)

        print(f'GAME OVER {score}')
        # keyboard.wait('esc')
