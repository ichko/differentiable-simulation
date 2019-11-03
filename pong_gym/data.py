import random
import numpy as np
import gym


def get_single_sequence(seq_len):
    env = gym.make('Pong-v0')
    env.reset()
    env.seed(0)

    actions, observations, rewards, done = [], [], [], []

    for _ in range(seq_len):
        # stay, up, down
        action = random.choice([1, 2, 3])  # env.action_space.sample()
        observation, reward, d, _info = env.step(action)

        one_hot_action = (np.array([1, 2, 3]) == action).astype(int)
        actions.append(one_hot_action)
        observations.append(observation)
        rewards.append(reward)
        done.append(d)

    return (np.array(actions),), (
        # Extract only one color channel
        (np.array(observations) / 255)[:, :, :, 1],
        np.array(rewards).reshape(-1, 1),
        np.array(done).reshape(-1, 1)
    )


def env_sequences_generator(seq_len):
    while True:
        yield get_single_sequence(seq_len)


if __name__ == '__main__':
    gen = env_sequences_generator(8)
    input, output = next(gen)
    print([o.shape for o in output])
