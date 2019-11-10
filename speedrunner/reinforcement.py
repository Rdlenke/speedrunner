import gc
import ray

import warnings

from datetime import datetime

@ray.remote(num_cpus=1, num_gpus=0.1, memory=1000 * 1024 * 1024)
class Reinforcement():
    def __init__(self, config):
        self.config = config

        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['KMP_WARNINGS'] = '0'

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=FutureWarning)
            import tensorflow as tf
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

            self.tf = tf
            
            from stable_baselines import DQN, A2C
            from stable_baselines.common.cmd_util import make_atari_env

            self.make_atari_env = make_atari_env
            self.DQN = DQN
            self.A2C = A2C

    def get_env(self, env_id):
        atari_env = self.make_atari_env(env_id, num_env=1, seed=datetime.now().microsecond)
        return atari_env 

    def create_dqn_model(self, env_id, lr=1e-4, df=0.99):
        dqn_hyperparameters = {
            'policy': 'CnnPolicy',
            'buffer_size': 10000,
            # 'learning_rate': 1e-4,
            'learning_starts': 10000,
            'target_network_update_freq': 1000,
            'train_freq': 4,
            'exploration_final_eps': 0.01,
            'exploration_fraction': 0.1,
            'prioritized_replay_alpha': 0.6,
            'prioritized_replay': True
        }
        
        dqn_env = self.get_env(env_id)
        dqn_model = self.DQN(env=dqn_env, learning_rate=lr, gamma=df, verbose=0, **dqn_hyperparameters)

        return dqn_model
    
    def create_a2c_model(self, env_id, lr=1e-4, gc=0.5 ,df=0.99):
        a2c_hyperparameters = {
        'policy': 'CnnPolicy',
        'lr_schedule': 'constant'
        }

        a2c_env = self.get_env(env_id)
        
        a2c_model = self.A2C(env=a2c_env, learning_rate=lr, max_grad_norm=gc, gamma=df,
                             verbose=0,
                             **a2c_hyperparameters)

        return a2c_model

    def evaluate(self, ind):
        if(len(ind) == 2):
            model = self.create_dqn_model('MsPacmanNoFrameskip-v4', ind[0], ind[1])
        else:
            model = self.create_a2c_model('MsPacmanNoFrameskip-v4', ind[0], ind[1], ind[2])

        model.learn(self.config['n_steps'])
        
        episode_rewards = []

        env = self.get_env('MsPacmanNoFrameskip-v4')
        
        for current in range(0, self.config['n_episodes']):
            reward_sum = 0
            done = False
            obs = env.reset()
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, _ = env.step(action)
                reward_sum += reward

            if done:
                env.reset()

            episode_rewards.append(reward_sum)

        
        model.graph.finalize()
        model.sess.close()
        self.tf.reset_default_graph()

        del model
        del env
        
        gc.collect()
        
        mean = sum(episode_rewards) / len(episode_rewards) 
        
        return mean, 
