from reinforcement import *

import logging
import matplotlib.pyplot as plt

logging.getLogger('speedrunner')

class Checkinpoint():
    best_ind = None
    best_model = None
    save_path = '.'
    config = None

    fitness_increment_data = {}

    def __init__(self, save_path, config):
        self.save_path = save_path
        self.config = config
        self.fitness_increment_data['gens'] = []
        self.fitness_increment_data['fitnesses'] = []

    def check_model(self, ind, gen):
        if(self.best_ind == None):
            self.best_ind = ind
        elif(ind.fitness.values[0] > self.best_ind.fitness.values[0]):
            logging.info(f'Best fitness is {ind.species} {ind.fitness.values[0]}. Saving...')
            self.fitness_increment_data['fitnesses'].append(ind.fitness.values[0])
            self.fitness_increment_data['gens'].append(gen)

            self.best_ind = ind

            self.save_model()

    def save_model(self):
        if(self.best_ind.species == 'dqn'):
            model = create_dqn_model('MsPacmanNoFrameskip-v4', self.best_ind[0], self.best_ind[1])
        else:
            model = create_a2c_model('MsPacmanNoFrameskip-v4', self.best_ind[0], self.best_ind[1],
                                     self.best_ind[2])
        model.save(self.save_path)

    def plot(self):
        plt.plot(self.fitness_increment_data['gens'], self.fitness_increment_data['fitnesses'])
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.savefig('plot.png')


