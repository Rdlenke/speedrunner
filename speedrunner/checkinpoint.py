from .reinforcement import *

import logging
import matplotlib.pyplot as plt

import numpy as np
import os
import pickle

logging.getLogger('speedrunner')

class Checkinpoint():
    best_ind = None
    best_model = None
    log_path = None
    config = None

    fitness_increment_data = {}

    dqn_params_data = {}
    a2c_params_data = {}

    def __init__(self, log_path, config):
        os.makedirs(log_path, exist_ok=True)
        self.log_path = log_path

        self.config = config
        self.fitness_increment_data['gens'] = []
        self.fitness_increment_data['fitnesses'] = []
        self.dqn_params_data['pop_learning_rate'] = []
        self.dqn_params_data['pop_gamma'] = []
        self.dqn_params_data['learning_rate'] = []
        self.dqn_params_data['gamma'] = []
        self.dqn_params_data['gens'] = []

        self.a2c_params_data['pop_learning_rate'] = []
        self.a2c_params_data['pop_gamma'] = []
        self.a2c_params_data['pop_max_grad_norm'] = []
        self.a2c_params_data['learning_rate'] = []
        self.a2c_params_data['gamma'] = []
        self.a2c_params_data['max_grad_norm'] = []
        self.a2c_params_data['gens'] = []

    def check_model(self, pop, gen):
        best_ind = pop[0]
        self.best_ind = best_ind

        self.fitness_increment_data['gens'].append(gen)
        self.fitness_increment_data['fitnesses'].append(best_ind.fitness.values[0])

        if len(best_ind) == 2:
            dqn_pop = [x for x in pop if len(x) == 2]

            pop_learning_rate = [x[0] for x in dqn_pop]
            pop_gamma = [x[1] for x in dqn_pop]

            self.dqn_params_data['pop_learning_rate'].append(pop_learning_rate)
            self.dqn_params_data['pop_gamma'].append(pop_gamma)
            self.dqn_params_data['learning_rate'].append(best_ind[0])
            self.dqn_params_data['gamma'].append(best_ind[1])
            self.dqn_params_data['gens'].append(gen)
        else:
            a2c_pop = [x for x in pop if len(x) == 3]

            pop_learning_rate = [x[0] for x in a2c_pop]
            pop_max_grad_norm = [x[1] for x in a2c_pop]
            pop_gamma = [x[2] for x in a2c_pop]

            self.a2c_params_data['pop_learning_rate'].append(pop_learning_rate)
            self.a2c_params_data['pop_max_grad_norm'].append(pop_max_grad_norm)
            self.a2c_params_data['pop_gamma'].append(pop_gamma)

            self.a2c_params_data['learning_rate'].append(best_ind[0])
            self.a2c_params_data['max_grad_norm'].append(best_ind[1])
            self.a2c_params_data['gamma'].append(best_ind[2])
            self.a2c_params_data['gens'].append(gen)

    def plot_best_param_evolution(self):
        with open('a2c_params_data.pickle', 'wb') as f:
            pickle.dump(self.a2c_params_data, f)

        with open('dqn_params_data.pickle', 'wb') as f:
            pickle.dump(self.dqn_params_data, f)

        if len(self.best_ind) == 2:
            self.dqn_params_data['gens'] = [x + 1 if index == 0 else x + 2 for index, x in enumerate(self.dqn_params_data['gens'])]
            
            x = [np.full(len(y), x) for x, y in zip(self.dqn_params_data['pop_learning_rate'], self.dqn_params_data['gens'])]
            x = np.hstack(x)
            labels = [f'Geração {i}' for i in np.unique(x)]

            plt.figure(figsize=(30, 30))

            plt.title('Evolução do Parâmetro learning rate para toda população')
            plt.scatter(x, self.dqn_params_data['pop_learning_rate'])
            plt.xticks(np.unique(x), labels, rotation='vertical')
            plt.ylabel('Learning Rate')

            plt.savefig(1, os.path.join(self.log_path, 'lr-pop-dqn.png'))

            x = [np.full(len(y), x) for x, y in zip(self.dqn_params_data['pop_gamma'], self.dqn_params_data['gens'])]
            x = np.hstack(x)

            plt.figure(figsize=(30, 30))

            plt.title('Evolução do Parâmetro Discount Factor para toda população')
            plt.scatter(x, self.dqn_params_data['pop_gamma'])
            plt.xticks(x, labels, rotation='vertical')
            plt.ylabel('Discount Factor')

            plt.savefig(1, os.path.join(self.log_path, 'gamma-pop-dqn.png'))

            labels = [f'Geração {x}' for x in self.dqn_params_data['gens']]
            x = np.arange(len(self.dqn_params_data['gens']) + 1)

            plt.figure(figsize=(30, 30))

            plt.title('Evolução do Parâmetro learning rate')
            plt.plot(self.dqn_params_data['learning_rate'])
            plt.xticks(x, labels, rotation='vertical')
            plt.ylabel('Learning Rate')

            plt.savefig(1, os.path.join(self.log_path, 'lr-dqn.png'))

            plt.figure(figsize=(30, 30))

            plt.title('Evolução do parâmetro discount factor')
            plt.plot(self.dqn_params_data['gamma'])
            plt.xticks(x, labels, rotation='vertical')
            plt.ylabel('Discount Rate')

            plt.savefig(os.path.join(self.log_path, 'gamma-dqn.png'))
        else:
            self.a2c_params_data['gens'] = [x + 1 if index == 0 else x + 2 for index, x in enumerate(self.a2c_params_data['gens'])]

            x = [np.full(len(y), x) for x, y in zip(self.a2c_params_data['gens'], self.a2c_params_data['pop_learning_rate'])]
            x = np.hstack(x)

            labels = [f'Geração {i}' for i in np.unique(x)]
            
            plt.figure(figsize=(30, 30))

            plt.title('Evolução do parâmetro learning rate para toda população')
            plt.scatter(x, np.hstack(self.a2c_params_data['pop_learning_rate']))
            plt.xticks(np.unique(x), labels, rotation='vertical')
            plt.ylabel('Learning Rate')

            plt.savefig(os.path.join(self.log_path, 'pop-lr-a2c.png'))

            x = [np.full(len(y), x) for x, y in zip(self.a2c_params_data['gens'], self.a2c_params_data['pop_max_grad_norm'])]
            x = np.hstack(x)

            plt.figure(figsize=(30, 30))

            plt.title('Evolução do parâmetro max grad norm para toda população')
            plt.scatter(x, np.hstack(self.a2c_params_data['pop_max_grad_norm']))
            plt.xticks(np.unique(x), labels, rotation='vertical')
            plt.ylabel('Max Grad Norm')

            plt.savefig(os.path.join(self.log_path, 'pop-max-grad-a2c.png'))

            x = [np.full(len(y), x) for x, y in zip(self.a2c_params_data['gens'], self.a2c_params_data['pop_gamma'])]
            x = np.hstack(x)
            
            plt.figure(figsize=(30, 30))

            plt.title('Evolução do parâmetro discount factor para toda população')
            plt.scatter(x, np.hstack(self.a2c_params_data['pop_gamma']))
            plt.xticks(np.unique(x), labels, rotation='vertical')
            plt.ylabel('Discount Factor')

            plt.savefig(os.path.join(self.log_path, 'pop-gamma-a2c.png'))

            x = np.arange(len(self.a2c_params_data['gens']) + 1)
            labels = [f'Geração {x}' for x in self.a2c_params_data['gens']]

            plt.figure(figsize=(30, 30))

            plt.title('Evolução do Parâmetro learning rate')
            plt.plot(self.a2c_params_data['learning_rate'])
            plt.xticks(x, labels, rotation='vertical')
            plt.ylabel('Learning Rate')

            plt.savefig(os.path.join(self.log_path, 'lr-a2c.png'))

            plt.figure(figsize=(30, 30))

            plt.title('Evolução do Parâmetro Max Grad Norm')
            plt.plot(self.a2c_params_data['max_grad_norm'])
            plt.xticks(x, labels, rotation='vertical')
            plt.ylabel('Max Grad Norm')

            plt.savefig(os.path.join(self.log_path, 'max-grad-a2c.png'))

            plt.figure(figsize=(30, 30))

            plt.title('Evolução do parâmetro discount factor')
            plt.plot(self.a2c_params_data['gamma'])
            plt.xticks(x, labels, rotation='vertical')
            plt.ylabel('Discount Rate')

            plt.savefig(os.path.join(self.log_path, 'gamma-a2c.png'))


    def plot_best(self):
        with open('general.pickle', 'wb') as f:
            pickle.dump(self.fitness_increment_data, f)

        if len(self.best_ind) == 2:
            print('O melhor indivíduo foi o DQN com os parâmetros:')
            print(f'Learning rate: {self.best_ind[0]}')
            print(f'Gamma: {self.best_ind[1]}')
        else:
            print('O melhor indivíduo foi o A2C com os parâmetros:')
            print(f'Learning rate: {self.best_ind[0]}')
            print(f'Max grad norm: {self.best_ind[1]}')
            print(f'Gamma: {self.best_ind[2]}')


        self.fitness_increment_data['gens'] = [x + 1 if index == 0 else x + 2 for index, x in enumerate(self.fitness_increment_data['gens'])]

        labels = [f'Geração {x}' for x in self.fitness_increment_data['gens']]

        plt.figure(figsize=(30, 30))

        plt.plot(self.fitness_increment_data['gens'], self.fitness_increment_data['fitnesses'])
        plt.title('Fitness do melhor indivíduo por geração')
        plt.xticks(np.arange(1, len(self.fitness_increment_data['gens']) + 1), labels, rotation='vertical')
        plt.ylabel('Fitness')

        plt.savefig(os.path.join(self.log_path, 'best.png'))

