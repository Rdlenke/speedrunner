from speedrunner.evolutionary import Evolutionary
from speedrunner.checkinpoint import Checkinpoint
from speedrunner.reinforcement import Reinforcement

import numpy as np

import ray
import gc
import pickle

ray.init(redis_address="desktopg02:6379")

config = {
    'n_generations': 150,
    'pop_size': 500,
    'n_steps': int(5000),
    'n_episodes': 5,
    'reintroduction_threshold': 4
}

dqn_params_data = {}

dqn_params_data['pop_learning_rate'] = []
dqn_params_data['pop_gamma'] = []
dqn_params_data['learning_rate'] = []
dqn_params_data['gamma'] = []
dqn_params_data['gens'] = []
dqn_params_data['fitnesses'] = []

a2c_params_data = {}

a2c_params_data['pop_learning_rate'] = []
a2c_params_data['pop_gamma'] = []
a2c_params_data['pop_max_grad_norm'] = []
a2c_params_data['fitnesses'] = []

a2c_params_data['learning_rate'] = []
a2c_params_data['gamma'] = []
a2c_params_data['max_grad_norm'] = []

fitness_increment_data = {}
fitness_increment_data['gens'] = []
fitness_increment_data['fitnesses'] = []
fitness_increment_data['species'] = []
fitness_increment_data['extinguished'] = []

def callback(pops, gen, extinguished, best_ind):
    global a2c_params_data, dqn_params_data, config, fitness_increment_data

    if gen == 0:
        dqn_pop = pops['dqn']

        pop_learning_rate = [x[0] for x in dqn_pop]

        pop_gamma = [x[1] for x in dqn_pop]

        dqn_params_data['pop_learning_rate'].append(pop_learning_rate)
        dqn_params_data['pop_gamma'].append(pop_gamma)

        a2c_pop = pops['a2c']

        pop_learning_rate = [x[0] for x in a2c_pop]
        pop_gamma = [x[1] for x in a2c_pop]
        pop_max_grad_norm = [x[2] for x in a2c_pop]

        a2c_params_data['pop_learning_rate'].append(pop_learning_rate)
        a2c_params_data['pop_gamma'].append(pop_gamma)
        a2c_params_data['pop_max_grad_norm'].append(pop_max_grad_norm)

    else:
        fitness_increment_data['gens'].append(gen)
        fitness_increment_data['fitnesses'].append(best_ind.fitness.values[0])
        fitness_increment_data['species'].append(best_ind.species)
        fitness_increment_data['extinguished'].append(extinguished)

        dqn_pop = pops['dqn']

        if dqn_pop != []:
            pop_fitnesses = [x.fitness.values[0] for x in dqn_pop]

            dqn_params_data['fitnesses'].append(pop_fitnesses)

            pop_learning_rate = [x[0] for x in dqn_pop]

            pop_gamma = [x[1] for x in dqn_pop]

            dqn_params_data['pop_learning_rate'].append(pop_learning_rate)
            dqn_params_data['pop_gamma'].append(pop_gamma)

            best_ind_dqn = dqn_pop[0]

            dqn_params_data['learning_rate'].append(best_ind_dqn[0])
            dqn_params_data['gamma'].append(best_ind_dqn[1])

        a2c_pop = pops['a2c']

        if a2c_pop != []:
            pop_fitnesses = [x.fitness.values[0] for x in a2c_pop]

            a2c_params_data['fitnesses'].append(pop_fitnesses)

            pop_learning_rate = [x[0] for x in a2c_pop]
            pop_gamma = [x[1] for x in a2c_pop]
            pop_max_grad_norm = [x[2] for x in a2c_pop]

            a2c_params_data['pop_learning_rate'].append(pop_learning_rate)
            a2c_params_data['pop_gamma'].append(pop_gamma)
            a2c_params_data['pop_max_grad_norm'].append(pop_max_grad_norm)

            best_ind_a2c = a2c_pop[0]

            a2c_params_data['learning_rate'].append(best_ind_a2c[0])
            a2c_params_data['gamma'].append(best_ind_a2c[1])
            a2c_params_data['max_grad_norm'].append(best_ind_a2c[2])

    if gen == (config['n_generations']):
        with open('a2c_params_data.pickle', 'wb') as f:
            pickle.dump(a2c_params_data, f)

        with open('dqn_params_data.pickle', 'wb') as f:
            pickle.dump(dqn_params_data, f)

        with open('general.pickle', 'wb') as f:
            pickle.dump(fitness_increment_data, f)

def main():
    targets = {'dqn': 2, 'a2c': 3}

    evolution = Evolutionary(targets, config=config, seed=69, callback=callback)

    evolution.create_pop('dqn')
    evolution.create_pop('a2c')

    evolution.register_evaluation_class(Reinforcement)

    evolution.run()

if __name__ == '__main__':
    main()
