# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython
import datetime
date = datetime.datetime.today()
import sys
sys.path.insert(0, "/home/ubuntu/work/codes/rl_openai/LuxPythonEnvGym")
print(sys.path)
import luxai2021
print(luxai2021.__file__)

sys.path.insert(0, "/home/ubuntu/work/codes/rl_openai/stable-baselines3")
print(sys.path)
import stable_baselines3
print(stable_baselines3.__file__)

# %% [markdown]
# # Lux AI Deep Reinforcement Learning Environment Example
# See https://github.com/glmcdona/LuxPythonEnvGym for environment project and updates.
# 
# This is a python replica of the Lux game engine to speed up training. It reformats the agent problem into making a action decision per-unit for the team.

# %%
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
print(device)


# %%
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
from rulebase_agent_policy import RulebaseAgentPolicy

from luxai2021.env.agent import Agent
from luxai2021.env.lux_env import LuxEnvironment
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from luxai2021.game.constants import LuxMatchConfigs_Default
import torch.nn as nn
import gym
import torch.nn.functional as F




# Default Lux configs
configs = LuxMatchConfigs_Default
configs["seed"] = 5621242

# # Create a default opponent agent
# load_path = "models_cnn_layer3"
# oppenent_pre_trained_PPO_model = PPO.load(f"/home/ubuntu/work/codes/rl_openai/models_2021-10-24 05:07:41.870598/model.zip")
# pre_trained_PPO_model.policy.load_state_dict(torch.load("/home/ubuntu/work/codes/rl_openai/model_cnn_lyaer3_state_dict.pth"), strict=False)
# pre_trained_PPO_model.policy.eval()
# opponent = RulebaseAgentPolicy(mode="inference")
# opponent = AgentPolicy(model=oppenent_pre_trained_PPO_model, mode="inference")
opponent = Agent()

# # Create a RL agent in training mode
# player_pre_trained_PPO_model = PPO.load(f"/home/ubuntu/work/codes/rl_openai/models_2021-10-27 11:40:03.307680/model.zip")
# pre_trained_PPO_model.policy.load_state_dict(torch.load("/home/ubuntu/work/codes/rl_openai/model_cnn_lyaer3_state_dict.pth"), strict=False)
# player = AgentPolicy(model=player_pre_trained_PPO_model, mode="train")
player = AgentPolicy(mode="train")
# player = Agent()

# Create the Lux environment
env = LuxEnvironment(configs=configs,
                     learning_agent=player,
                     opponent_agent=opponent)

class BasicConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim, 
            kernel_size=kernel_size, 
            padding=(kernel_size[0] // 2, kernel_size[1] // 2)
        )
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        return h


class LuxNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(LuxNet, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        layers, filters = 3, 32
        self.conv0 = BasicConv2d(n_input_channels, filters, (3, 3), True)
        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, features_dim, bias=False)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        p = self.head_p(h_head)
        return p

policy_kwargs = dict(
    features_extractor_class=LuxNet,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=[]
)

import datetime
date = datetime.datetime.today()
path = f"models_{date}"
# Define the model, you can pick other RL algos from Stable Baselines3 instead if you like
model = PPO("MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=f"./lux_tensorboard_{path}/",
                learning_rate=0.001,
                gamma=0.999,
                gae_lambda=0.95,
                batch_size=4096,
                n_steps=2048 * 8,
                policy_kwargs=policy_kwargs
                            )
# Define a learning rate schedule
# (number of steps, learning_rate)
schedule = [
    # (100, 0.01)
    (20000000, 0.01),
    # (2000000, 0.001),
    # (2000000, 0.0001),
]


# %%
model.policy.load_state_dict(torch.load("/home/ubuntu/work/codes/rl_openai/model_cnn_lyaer3_state_dict.pth"), strict=False)
# model.policy = player_pre_trained_PPO_model.policy

# %%
from stable_baselines3.common.utils import get_schedule_fn

print("Training model...")
run_id = 1
# Save a checkpoint every 1M steps
checkpoint_callback = CheckpointCallback(save_freq=1000000,
                                         save_path=f'./{path}/',
                                         name_prefix=f'rl_model_{run_id}')

# Train the policy
for steps, learning_rate in schedule:
    model.lr_schedule = get_schedule_fn(learning_rate)
    model.learn(total_timesteps=steps,
                callback=checkpoint_callback,
                reset_num_timesteps = False)

# Save final model
model.save(path=f'{path}/model.zip')
torch.save(model.policy.state_dict(), f'{path}/model_state_dict.pth')

print("Done training model.")

# %% [markdown]
# # Set up a Kaggle Submission and lux replay environment for the agent

# %%
"""
This downloads two required python package dependencies that are not pre-installed
by Kaggle yet.

This places the following two packages in the current working directory:
    luxai2021
    stable_baselines3
"""

import os
import shutil
import subprocess
import tempfile

def localize_package(git, branch, folder):
    if os.path.exists(folder):
        print("Already localized %s" % folder)
    else:
        # https://stackoverflow.com/questions/51239168/how-to-download-single-file-from-a-git-repository-using-python
        # Create temporary dir
        t = tempfile.mkdtemp()

        args = ['git', 'clone', '--depth=1', git, t, '-b', branch]
        res = subprocess.Popen(args, stdout=subprocess.PIPE)
        output, _error = res.communicate()

        if not _error:
            print(output)
        else:
            print(_error)
        
        # Copy desired file from temporary dir
        shutil.move(os.path.join(t, folder), '.')
        # Remove temporary dir
        shutil.rmtree(t, ignore_errors=True)

localize_package('https://github.com/glmcdona/LuxPythonEnvGym.git', 'main', 'luxai2021')
localize_package('https://github.com/glmcdona/LuxPythonEnvGym.git', 'main', 'kaggle_submissions')
localize_package('https://github.com/DLR-RM/stable-baselines3.git', 'master', 'stable_baselines3')


# %%
# Move the dependent packages into kaggle submissions
get_ipython().system('mv luxai2021 kaggle_submissions')
get_ipython().system('mv stable_baselines3 kaggle_submissions')
get_ipython().system('rm ./kaggle_submissions/agent_policy.py')
get_ipython().system('cp agent_policy.py kaggle_submissions')

# Copy the agent and model to the submission 
get_ipython().system('cp ./agent_policy.py kaggle_submissions')
get_ipython().system(f'cp ./"models_2021-10-19 16:22:19.300762"/rl_model_1_100000_steps.zip kaggle_submissions')

get_ipython().system('ls kaggle_submissions')


# %%
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import numpy
from kaggle_environments import make
import json
# run another match but with our empty agent
env = make("lux_ai_2021", configuration={"seed": 5621242, "loglevel": 2, "annotations": True}, debug=True)

# Play the environment where the RL agent plays against itself
steps = env.run(["./kaggle_submissions/main.py", "simple_agent"])
print("done")

# %%
# Render the match
env.render(mode="ipython", width=1200, height=800)

# %% [markdown]
# # Prepare and submit the kaggle submission

# %%
get_ipython().system('tar -czf submission.tar.gz -C kaggle_submissions .')
get_ipython().system('ls')

# %% [markdown]
# 

