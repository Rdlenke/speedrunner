from deap import base
from deap import tools
from deap import creator

import dask.bag as db

import numpy as np

def dask_map(func, iterable):
    bag = db.from_sequence(iterable).map(func)
    return bag.compute()

class Evolutionary():
    toolbox = None
    maximizing = True
    config = {}
    targets = {}
    pops = {}

    def __init__(self, targets, maximizing, config):
        self.toolbox = base.Toolbox()
        self.toolbox.register('random', np.random.uniform, 1e-6, 1)
        self.toolbox.register('map', dask_map)

        self.maximizing = maximizing
        self.config = config
        self.targets = targets

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
        self.toolbox.register('mutate', tools.mutESLogNormal, c=1.0, indpb=0.33)

        # self.toolbox.register('mate', checkStrategy(0.001))
        # self.toolbox.register('mutate', checkStrategy(0.001))


    def __create_individual__(self, name, number_of_params):
        def generate():
            ind_creator = getattr(creator, 'Individual' + name)
            ind = ind_creator(self.toolbox.random() for _ in range(number_of_params))

            ind_strategy_creator = getattr(creator, 'Strategy' + name)
            ind.strategy = ind_strategy_creator(self.toolbox.random() for _ in range(number_of_params))

            return ind

        self.toolbox.register('generate_' + name, generate)

    def create_pop(self, name):
        pop_generator = getattr(self.toolbox, 'pop_' + name)
        self.pops[name] = pop_generator(int(self.config['pop_size'] / 2))

    def __create_pop__(self, name):
        ind_generator = getattr(self.toolbox, 'generate_' + name)
        self.toolbox.register('pop_' + name, tools.initRepeat, list, ind_generator)

    def register_evaluation_function(self, func):
        self.toolbox.register('evaluate', func)

    def select(self):
        keys = list(self.targets.keys())

        print(f'Keys: {keys}')
        print(f'Pops: {self.pops}')

        pop_one = self.pops[keys[0]]
        pop_two = self.pops[keys[1]]

        if(pop_one != []):
            fitnesses = self.toolbox.map(self.toolbox.evaluate, pop_one)

            for ind, fitness in zip(pop_one, fitnesses):
                ind.fitness.values = fitness

        if(pop_two != []):
            fitnesses = self.toolbox.map(self.toolbox.evaluate, pop_two)

            for ind, fitness in zip(pop_two, fitnesses):
                ind.fitness.values = fitness

        pop = [*pop_one, *pop_two]
        pop = tools.selBest(pop, k=self.config['pop_size'])
        pop_one = [x for x in pop if len(x) == self.targets[keys[0]]]
        pop_two = [x for x in pop if len(x) == self.targets[keys[1]]]

        self.pops[keys[0]] = pop_one
        self.pops[keys[1]] = pop_two

    def mutate(self):
        print(f'Pop: {list(self.pops.values())}')
        pop = list(self.pops.values())[0]

        print(f'Pop before mutation {self.pops}')

        for ind in pop:
            print(f'Ind: {ind}')
            ind = self.toolbox.mutate(ind)[0]

        print(f'Pop before mutation {self.pops}')

    def __single_pop_crossover__(self, name):
        childs = []

        pop = self.pops[name]

        for parent_1, parent_2 in zip(pop[::2], pop[1::2]):
            child = self.toolbox.mate(parent_1, parent_2)[0]

            childs.append(child)

        self.pops[name].append(childs)

    def crossover(self):
        keys = list(self.pops.keys())

        for key in keys:
            self.__single_pop_crossover__(key)

