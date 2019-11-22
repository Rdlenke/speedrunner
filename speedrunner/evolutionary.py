from deap import base
from deap import tools
from deap import creator

import itertools

import numpy as np

import ray

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
    evaluator = None
    random = None

    extinguished = {}

    def __init__(self, targets, config, seed=69, callback=None):
        self.random = np.random.RandomState(seed)
        self.toolbox = base.Toolbox()
        self.toolbox.register('random', self.random.uniform, 1e-6, 1)
        self.toolbox.register('reset_seed', self.random.seed, seed)

        self.config = config
        self.targets = targets

        self.checkinpoint = Checkinpoint(config, callback)

        creator.create('FitnessMax', base.Fitness, weights=(1.0,))

        for species in targets.keys():
            self.__create_structures__(species)
            self.__create_individual__(species, targets[species])
            self.__create_pop__(species)

    def __create_structures__(self, name):
        creator.create('Individual' + name, list, fitness=creator.FitnessMax,
                        species=name, strategy=None)

        creator.create('Strategy' + name, list)

        def cxIntermediary(ind1, ind2):
            ind3 = base.Toolbox().clone(ind1)

            for index, (gen_ind1, gen_ind2) in enumerate(zip(ind1, ind2)):
                ind3[index] = ((gen_ind1 + gen_ind2) / 2.0)

            del ind3.fitness.values

            return ind3,

        self.toolbox.register('select', tools.selBest)

        self.toolbox.register('mate', cxIntermediary)
        self.toolbox.register('mutate', tools.mutESLogNormal, c=1.0, indpb=0.5)

    def __create_individual__(self, name, number_of_params):
        def generate():
            ind_creator = getattr(creator, 'Individual' + name)
            ind = ind_creator(self.toolbox.random() for _ in range(number_of_params))

            ind_strategy_creator = getattr(creator, 'Strategy' + name)
            ind.strategy = ind_strategy_creator(self.toolbox.random() for _ in range(number_of_params))
            del ind.fitness.values

            self.toolbox.reset_seed()
            return ind

        self.toolbox.register('generate_' + name, generate)

    def create_pop(self, name):
        pop_generator = getattr(self.toolbox, 'pop_' + name)
        self.pops[name] = pop_generator(int(self.config['pop_size'] / len(self.targets.keys())))


    def __create_pop__(self, name):
        ind_generator = getattr(self.toolbox, 'generate_' + name)
        self.toolbox.register('pop_' + name, tools.initRepeat, list, ind_generator)

    def register_evaluation_class(self, func):
        self.toolbox.register('actor', func.remote)


    def __evaluate_individuals__(self):
        keys = list(self.targets.keys())

        for key in keys:
            pop = [x for x in self.pops[key] if not x.fitness.values]

            if(pop != []):
                fitnesses = []

                for ind in pop:
                    evaluator = self.toolbox.actor(self.config)
                    fitnesses.append(evaluator.evaluate.remote(list(ind)))

                fitnesses = ray.get(fitnesses)

                for ind, fitness in zip(pop, fitnesses):
                    ind.fitness.values = fitness

                del evaluator
                del fitnesses


    def __select__(self):
        self.__evaluate_individuals__()

        keys = list(self.pops.keys())

        pop = [self.pops[key] for key in keys]

        pop = list(itertools.chain(*pop))

        pop = tools.selBest(pop, k=self.config['pop_size'])

        for key in keys:
            self.pops[key] = [x for x in pop if x.species == key]


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

        extinguished_species = []

        for species in keys:
            if self.pops[species] == []:
                extinguished_species.append(species)

        return extinguished_species

    def __check_reintroduction__(self):
        if self.current_gen <= self.config['reintroduction_threshold']:
            result = self.__check_if_population_is_null__()

            if result != []:
                self.extinguished.update({ self.current_gen: result })
                self.__reintroduce__(result)

    def __reintroduce__(self, species):
        list(map(self.create_pop, species))


    def run(self):
        self.__select__()

        for gen in tqdm(range(self.config['n_generations'])):
            self.current_gen = gen
            self.__check_reintroduction__()
            self.__crossover__()
            self.__mutate__()
            self.__select__()
            self.checkinpoint.check_model(self.pops, self.current_gen, self.extinguished)
            self.extinguished = {}

        self.checkinpoint.show_best_model()

