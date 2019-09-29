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

    def check_model(self, pop, gen):
        best_ind = pop[0]

        self.fitness_increment_data['gens'].append(gen)
        self.fitness_increment_data['fitnesses'].append(best_ind.fitness.values[0])

    def plot_best(self, path):
        plt.plot(self.fitness_increment_data['gens'], self.fitness_increment_data['fitnesses'])
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.savefig(path)

