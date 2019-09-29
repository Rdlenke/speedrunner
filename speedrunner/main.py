from evolutionary import Evolutionary
from checkinpoint import Checkinpoint
from reinforcement import *

from functools import partialmethod

import numpy as np

config = {
    'n_generations': 20,
    'pop_size': 20,
    'n_steps': int(10000),
    'n_episodes': 3,
    'reintroduction_threshold': 4
}

def main():
    targets = {'dqn': 2, 'a2c': 3}

    checkinpoint = Checkinpoint('.', config)

    evolution = Evolutionary(targets, True, config, checkinpoint=checkinpoint)

    evolution.create_pop('dqn')
    evolution.create_pop('a2c')

    evolution.register_evaluation_function(evaluate)

    evolution.run()
    checkinpoint.plot()

def evaluate(ind):
    """
    Evaluation method
    """
    print('Evaluating...')

    print(f'Individual being evaluated: {ind}')

    if(len(ind) == 2):
        print('Create DQN model')
        model = create_dqn_model('MsPacmanNoFrameskip-v4', ind[0], ind[1])
    else:
        print('Create A2C model')
        model = create_a2c_model('MsPacmanNoFrameskip-v4', ind[0], ind[1], ind[2])

    env = get_env('MsPacmanNoFrameskip-v4')

    print(f'Created model')

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
