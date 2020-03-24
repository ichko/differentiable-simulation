import models
import util

import numpy as np
import gym

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    num_actions = env.action_space.n
    obs_size = np.prod(env.observation_space.shape)

    agent = models.DQNAgent(obs_size, num_actions)
    agent = util.make_persisted_model(agent, 'dqn_agent.pkl')

    # util.play_env(
    #     env,
    #     lambda _: env.action_space.sample(),
    #     duration=80,
    # )

    util.optimize(
        model=agent,
        next_batch=util.get_experience_generator(env, agent, bs=32),
        its=5000,
        optim_args=dict(lr=0.001),
        on_it_end=lambda i: agent.persist(),
    )
