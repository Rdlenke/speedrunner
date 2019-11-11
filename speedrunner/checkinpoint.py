from .reinforcement import *

import logging
import matplotlib.pyplot as plt

logging.getLogger('speedrunner')

class Checkinpoint():
    best_ind = None
    best_model = None
    save_path = '.'
    config = None

    fitness_increment_data = {}

    dqn_params_data = {}
    a2c_params_data = {}

    def __init__(self, save_path, config):
        self.save_path = save_path
        self.config = config
        self.fitness_increment_data['gens'] = []
        self.fitness_increment_data['fitnesses'] = []
        self.dqn_params_data['learning_rate'] = []
        self.dqn_params_data['gamma'] = []
        self.dqn_params_data['gens'] = []
        
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
            self.dqn_params_data['learning_rate'].append(best_ind[0])
            self.dqn_params_data['gamma'].append(best_ind[1])
            self.dqn_params_data['gens'].append(gen)
        else:
            self.a2c_params_data['learning_rate'].append(best_ind[0])
            self.a2c_params_data['max_grad_norm'].append(best_ind[1])
            self.a2c_params_data['gamma'].append(best_ind[2])
            self.a2c_params_data['gens'].append(gen)

    def plot_best_param_evolution(self):
        if len(self.best_ind) == 2:
            labels = [f'Geração {x}' for x in self.dqn_params_data['gens']]

            plt.title('Evolução do Parâmetro learning rate')
            plt.plot(y=self.dqn_params_data['learning_rate'])
            plt.xticks(np.arange(len(self.dqn_params_data['gens'])), labels)
            plt.xlabel('Gerações')
            plt.ylabel('Learning Rate')

            plt.savefig('lr-dqn.png')

            plt.figure()

            plt.title('Evolução do parâmetro discount factor')
            plt.xticks(np.arange(len(self.dqn_params_data['gens'])), labels)
            plt.plot(y=self.dqn_params_data['gamma'])
            plt.xlabel('Gerações')
            plt.ylabel('Discount Rate')

            plt.savefig('gamma-dqn.png')
        else:
            labels = [f'Geração {x}' for x in self.a2c_params_data['gens']]

            plt.title('Evolução do Parâmetro learning rate')
            plt.xticks(np.arange(len(self.a2c_params_data['gens'])), labels)
            plt.plot(y=self.a2c_params_data['learning_rate'])
            plt.xlabel('Gerações')
            plt.ylabel('Learning Rate')

            plt.savefig('lr-a2c.png')

            plt.figure()

            plt.title('Evolução do Parâmetro Max Grad Norm')
            plt.xticks(np.arange(len(self.a2c_params_data['gens'])), labels)
            plt.plot(y=self.a2c_params_data['max_grad_norm'])
            plt.xlabel('Gerações')
            plt.ylabel('Max Grad Norm')

            plt.savefig('max-grad-a2c.png')

            plt.figure()

            plt.title('Evolução do parâmetro discount factor')
            plt.xticks(np.arange(len(self.a2c_params_data['gens'])), labels)
            plt.plot(y=self.a2c_params_data['gamma'])
            plt.xlabel('Gerações')
            plt.ylabel('Discount Rate')

            plt.savefig('gamma-a2c.png')

    def plot_best(self, path):
        if len(self.best_ind) == 2:
            print('O melhor indivíduo foi o DQN com os parâmetros:')
            print(f'Learning rate: {self.best_ind[0]}')
            print(f'Gamma: {self.best_ind[1]}')
        else:
            print('O melhor indivíduo foi o A2C com os parâmetros:')
            print(f'Learning rate: {self.best_ind[0]}')
            print(f'Max grad norm: {self.best_ind[1]}')
            print(f'Gamma: {self.best_ind[2]}')

        plt.plot(self.fitness_increment_data['gens'], self.fitness_increment_data['fitnesses'])
        plt.title('Fitness do melhor indivíduo por geração')
        plt.xlabel('Gerações')
        plt.ylabel('Fitness')
        plt.savefig(path)

