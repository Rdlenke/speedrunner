from evolutionary import Evolutionary
from reinforcement import *

from functools import partialmethod

import numpy as np

config = {
    'n_generations': 100,
    'pop_size': 4,
    'n_steps': int(1000),
    'n_episodes': 5
}

def main():
    targets = {'dqn': 2, 'a2c': 3}

    evolution = Evolutionary(targets, True, config)

    evolution.create_pop('dqn')
    evolution.create_pop('a2c')

    evolution.register_evaluation_function(evaluate)

    evolution.select()

    for gen in range(config['n_generations']):
        evolution.crossover()
        evolution.mutate()
        evolution.select()

def evaluate(ind):
    """
    Evaluation method
    """
    print('Evaluating...')

    if ind.species == 'dqn':
        model, env = create_dqn_model('MsPacmanNoFrameskip-v4', ind[0], ind[1])
    else:
        model, env = create_a2c_model('MsPacmanNoFrameskip-v4', ind[0], ind[1], ind[2])

    env.reset()

    model.learn(config['n_steps'])

    episode_rewards = []

    for current in range(0, config['n_episodes']):
        print(f'Episode: {current}')
        reward_sum = 0
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward

        episode_rewards.append(reward_sum)

    return (np.mean(episode_rewards),)



if __name__ == '__main__':
    main()
