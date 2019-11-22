from .reinforcement import *

import logging
import os
import itertools

class Checkinpoint():
    best_ind = None
    config = None

    def __init__(self, config, callback):
        self.config = config
        self.run_callback = callback

    def check_model(self, pops, gen, extinguished):
        pop = list(pops.values())
        pop = list(itertools.chain(*pop))

        if gen != 0 :
            pop = sorted(pop, key=lambda x: x.fitness.values[0], reverse=True)

            if self.best_ind is None:
                self.best_ind = pop[0]
            elif pop[0].fitness.values[0] > self.best_ind.fitness.values[0]:
                self.best_ind = pop[0]

        if self.run_callback is not None:
            self.run_callback(pops, gen, extinguished, self.best_ind)

    def show_best_model(self):
        print(f'O melhor indivíduo foi o da espécie: {self.best_ind.species}')
        print(f'Com a fitness de incríveis: {self.best_ind.fitness.values[0]}')
        print(f'Com os parâmetros: {self.best_ind}')
