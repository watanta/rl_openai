import sys
sys.path.insert(0, "/home/ubuntu/work/codes/rl_openai/kaggle-environments/kaggle_environments")
print(sys.path)
import kaggle_environments
print(kaggle_environments.__file__)

sys.path.insert(0, "/home/ubuntu/work/codes/rl_openai/LuxPythonEnvGym")
print(sys.path)
import luxai2021
print(luxai2021.__file__)

import argparse
import glob
import os
import random
from typing import Callable

from stable_baselines3 import PPO  # pip install stable-baselines3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from importlib import reload
import agent_policy
# reload(agent_policy) # Reload the file from disk incase the above agent-writing cell block was edited
from agent_policy import AgentPolicy

from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
print(device)

# Default Lux configs
configs = LuxMatchConfigs_Default

# Create a default opponent agent
opponent = AgentWithModel(model="./opponent_model.zip")

# Create a RL agent in training mode
player = AgentPolicy(mode="train")

# Create the Lux environment
env = LuxEnvironment(configs=configs,
                     learning_agent=player,
                     opponent_agent=opponent)

# Define the model, you can pick other RL algos from Stable Baselines3 instead if you like
model = PPO("MlpPolicy",
                env,
                verbose=1,
                tensorboard_log="./lux_tensorboard/",
                learning_rate=0.001,
                gamma=0.999,
                gae_lambda=0.95,
                batch_size=4096,
                n_steps=2048 * 8
            )

# Define a learning rate schedule
# (number of steps, learning_rate)
schedule = [
    #(2000000, 0.01),
    (5000000, 0.001),
    (5000000, 0.0001),
]

from stable_baselines3.common.utils import get_schedule_fn

print("Training model...")
run_id = 1

# Save a checkpoint every 1M steps
checkpoint_callback = CheckpointCallback(save_freq=1000000,
                                         save_path='./models/',
                                         name_prefix=f'rl_model_{run_id}')

# Train the policy
for steps, learning_rate in schedule:
    model.lr_schedule = get_schedule_fn(learning_rate)
    model.learn(total_timesteps=steps,
                callback=checkpoint_callback,
                reset_num_timesteps = False)

# Save final model
model.save(path=f'models/model.zip')

print("Done training model.")