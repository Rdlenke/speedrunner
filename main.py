from speedrunner.evolutionary import Evolutionary
from speedrunner.checkinpoint import Checkinpoint
from speedrunner.reinforcement import Reinforcement 

import numpy as np

import ray
import gc
import pickle

ray.init(redis_address="desktopg02:6379")

config = {
    'n_generations': 50,
    'pop_size': 500,
    'n_steps': int(5000),
    'n_episodes': 3,
    'reintroduction_threshold': 4
}

def main():
    targets = {'dqn': 2, 'a2c': 3}

    checkinpoint = Checkinpoint('graph/', config)

    evolution = Evolutionary(targets, True, config, checkinpoint=checkinpoint)

    evolution.create_pop('dqn')
    evolution.create_pop('a2c')

    evolution.register_evaluation_function(Reinforcement)

    evolution.run()
    checkinpoint.plot_best()
    checkinpoint.plot_best_param_evolution()
   
    with open('checkinpoint.pickle', 'wb') as f:
        pickle.dump(checkinpoint, f)

if __name__ == '__main__':
    main()
