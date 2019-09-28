from stable_baselines.common.vec_env import VecFrameStack,DummyVecEnv, VecVideoRecorder
from stable_baselines.bench import Monitor
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines import DQN, A2C

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def create_dqn_model(env_id, lr=1e-4, df=0.99):
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

  dqn_env = make_atari_env(env_id, num_env=1, seed=0)
  dqn_model = DQN(env=dqn_env, learning_rate=lr,
                  gamma=df, verbose=1, **dqn_hyperparameters)

  return dqn_model, dqn_env

def create_a2c_model(env_id, lr=1e-4, gc=0.5 ,df=0.99):
  a2c_hyperparameters = {
  'policy': 'CnnPolicy',
  'lr_schedule': 'constant'
  }

  # The original number of envs is 16
  # Need to check the impact of more environment in the training

  a2c_env = make_atari_env(env_id, num_env=1, seed=0)
  #a2c_env = VecFrameStack(a2c_env, n_stack=4)
  a2c_model = A2C(env=a2c_env, learning_rate=lr, max_grad_norm=gc, gamma=df,
                  **a2c_hyperparameters, verbose=1)

  return a2c_model, a2c_env