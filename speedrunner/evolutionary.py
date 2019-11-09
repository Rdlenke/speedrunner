from deap import base
from deap import tools
from deap import creator

import dask
import numpy as np
import ray

import logging

logging.getLogger('speedrunner')

from .checkinpoint import Checkinpoint
from tqdm import tqdm
import gc

class Evolutionary():
    toolbox = None
    maximizing = True
    config = {}
    targets = {}
    pops = {}
    checkinpoint = None
    current_gen = 0

    def __init__(self, targets, maximizing, config, checkinpoint):
        self.toolbox = base.Toolbox()
        self.toolbox.register('random', np.random.uniform, 1e-6, 1)

        self.maximizing = maximizing
        self.config = config
        self.targets = targets

        self.checkinpoint = checkinpoint

        if maximizing:
            creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        else:
            creator.create('FitnessMin', base.Fitness, weights=(-1.0,))

        if targets is None:
            raise Exception('Empty list of population names!')

        for species in targets.keys():
            self.__create_structures__(species)
            self.__create_individual__(species, targets[species])
            self.__create_pop__(species)

    def __create_structures__(self, name):
        if(self.maximizing):
            creator.create('Individual' + name, list, fitness=creator.FitnessMax,
                           species=name, strategy=None)
        else:
            creator.create('Individual' + name, list, fitness=creator.FitnessMin,
                           species=name, strategy=None)

        creator.create('Strategy' + name, list)

        def cxIntermediary(ind1, ind2):
            ind3 = base.Toolbox().clone(ind1)

            for index, (gen_ind1, gen_ind2) in enumerate(zip(ind1, ind2)):
                ind3[index] = ((gen_ind1 + gen_ind2) / 2.0)

            del ind3.fitness.values

            return ind3,

        def checkStrategy(minstrategy):
            def decorator(func):
                def wrapper(*args, **kargs):
                    children = func(*args, **kargs)

                    for child in children:
                        for i, s in enumerate(child.strategy):
                            if s < minstrategy:
                                child.strategy[i] = minstrategy
                    return children
                return wrapper
            return decorator

        self.toolbox.register('select', tools.selBest)

        self.toolbox.register('mate', cxIntermediary)
        self.toolbox.register('mutate', tools.mutESLogNormal, c=1.0, indpb=0.5)

        # self.toolbox.register('mate', checkStrategy(0.001))
        # self.toolbox.register('mutate', checkStrategy(0.001))


    def __create_individual__(self, name, number_of_params):
        def generate():
            ind_creator = getattr(creator, 'Individual' + name)
            ind = ind_creator(self.toolbox.random() for _ in range(number_of_params))

            ind_strategy_creator = getattr(creator, 'Strategy' + name)
            ind.strategy = ind_strategy_creator(self.toolbox.random() for _ in range(number_of_params))
            del ind.fitness.values

            return ind

        self.toolbox.register('generate_' + name, generate)

    def create_pop(self, name):
        pop_generator = getattr(self.toolbox, 'pop_' + name)
        self.pops[name] = pop_generator(int(self.config['pop_size'] / 2))

        logging.debug(f'Population for {name}: {self.pops[name]} created.')


    def __create_pop__(self, name):
        ind_generator = getattr(self.toolbox, 'generate_' + name)
        self.toolbox.register('pop_' + name, tools.initRepeat, list, ind_generator)

    def register_evaluation_function(self, func):
        self.toolbox.register('evaluate', func)

    def __select__(self):
        keys = list(self.targets.keys())

        pop_one = [x for x in self.pops[keys[0]] if not x.fitness.values]
        pop_two = [x for x in self.pops[keys[1]] if not x.fitness.values]

        if(pop_one != []):
            fitnesses = []

            for ind in pop_one:
                fitnesses.append(self.toolbox.evaluate.remote(list(ind)))

            fitnesses = ray.get(fitnesses)

            for ind, fitness in zip(pop_one, fitnesses):
                ind.fitness.values = fitness

            del fitnesses

        if(pop_two != []):
            fitnesses = []

            for ind in pop_two:
                fitnesses.append(self.toolbox.evaluate.remote(list(ind)))

    
            fitnesses = ray.get(fitnesses)
    
            for ind, fitness in zip(pop_two, fitnesses):
                ind.fitness.values = fitness
    
            del fitnesses
    

        gc.collect()

        pop = [*self.pops[keys[0]], *self.pops[keys[1]]]
        pop = tools.selBest(pop, k=self.config['pop_size'])

        if(self.checkinpoint is not None):
           self.checkinpoint.check_model(pop, self.current_gen)

        pop_one = [x for x in pop if len(x) == self.targets[keys[0]]]
        pop_two = [x for x in pop if len(x) == self.targets[keys[1]]]

        self.pops[keys[0]] = pop_one
        self.pops[keys[1]] = pop_two

        del pop_one
        del pop_two

        gc.collect()


    def __mutate__(self):
        pop = list(self.pops.values())[0]

        for ind in pop:
            ind = self.toolbox.mutate(ind)[0]
            del ind.fitness.values


    def __single_pop_crossover__(self, name):

        pop = self.pops[name]

        for parent_1, parent_2 in zip(pop[::2], pop[1::2]):
            child = self.toolbox.mate(parent_1, parent_2)[0]

            self.pops[name].append(child)

    def __crossover__(self):
        keys = list(self.pops.keys())

        for key in keys:
            self.__single_pop_crossover__(key)

    def __check_if_population_is_null__(self):
        keys = list(self.pops.keys())
        if self.pops[keys[0]] == []:
            return str(keys[0])
        elif self.pops[keys[1]] == []:
            return str(keys[1])
        else:
            return ''

    def __check_reintroduction__(self):
        if self.current_gen <= self.config['reintroduction_threshold']:
            result = self.__check_if_population_is_null__()

            if result != '':
                self.__reintroduce__(result)

    def __reintroduce__(self, name):
        self.create_pop(name)


    def run(self):
        self.__select__()

        for gen in tqdm(range(self.config['n_generations'])):
            logging.info(f'Generation {gen}.')
            self.current_gen = gen
            self.__check_reintroduction__()
            self.__crossover__()
            self.__mutate__()
            self.__select__()

