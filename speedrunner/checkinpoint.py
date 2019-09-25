from reinforcement import *


class Checkinpoint():
    best_ind = None
    best_model = None
    save_path = '.'
    config = None

    def __init__(self, save_path, config):
        self.save_path = save_path
        self.config = config

    def check_model(self, ind):
        if(self.best_ind == None):
            self.best_ind = ind
        elif(ind.fitness.values[0] > self.best_ind.fitness.values[0]):
            print(f'Best fitness is {ind.species} {ind.fitness.values[0]}. Saving...')
            self.best_ind = ind

            self.save_model()

    def save_model(self):
        if(self.best_ind.species == 'dqn'):
            model, _ = create_dqn_model('MsPacmanNoFrameskip-v4', self.best_ind[0], self.best_ind[1])
        else:
            model, _ = create_a2c_model('MsPacmanNoFrameskip-v4', self.best_ind[0], self.best_ind[1],
                                     self.best_ind[2])
        model.save(self.save_path)


