import sys

from torch._C import ThroughputBenchmark
print(sys.version)

import numpy as np
import json
from pathlib import Path
import os
import random
from tqdm.notebook import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from imitation.data import types
from tqdm import tqdm

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def to_label(action):
    strs = action.split(' ')
    if strs[0] == "m" or strs[0] == "bcity": # if worker or cart
        unit_id = strs[1]
        if strs[0] == 'm':
            label = {'c': None, 'n': 0, 's': 1, 'w': 2, 'e': 3}[strs[2]]
        elif strs[0] == 'bcity':
            label = 4
    elif strs[0] == "bw" or strs[0] == "bc" or strs[0] == "r": # if citytile
        unit_id = strs[1]+" "+strs[2] # citytileはidを持たないので、位置を入れて自分を判別
        if strs[0] == "r":
            label = 0
        elif strs[0] == "bw":
            label = 1
        elif strs[0] == "bc":
            label = 2
    else: # transfer
        unit_id = None
        label = None
    return unit_id, label



def depleted_resources(obs):
    for u in obs['updates']:
        if u.split(' ')[0] == 'r':
            return False
    return True


def create_dataset_from_json(episode_dir, team_name='Toad Brigade'): 
    obses = {}
    samples = []
    append = samples.append
    
    episodes = [path for path in Path(episode_dir).glob('*.json') if 'output' not in path.name]
    for filepath in tqdm(episodes): 
        with open(filepath) as f:
            json_load = json.load(f)

        ep_id = json_load['info']['EpisodeId']
        index = np.argmax([r or 0 for r in json_load['rewards']])
        if json_load['info']['TeamNames'][index] != team_name:
            continue

        for i in range(len(json_load['steps'])-1):
            if json_load['steps'][i][index]['status'] == 'ACTIVE':
                actions = json_load['steps'][i+1][index]['action']
                obs = json_load['steps'][i][0]['observation']
                
                if depleted_resources(obs):
                    break
                
                obs['player'] = index
                obs = dict([
                    (k,v) for k,v in obs.items() 
                    if k in ['step', 'updates', 'player', 'width', 'height']
                ])
                obs_id = f'{ep_id}_{i}'
                obses[obs_id] = obs
                                
                for action in actions:
                    unit_id, label = to_label(action)
                    if label is not None: # move c or transfer 
                        append((obs_id, unit_id, label))

    return obses, samples

def make_input(obs, unit_id):
    width, height = obs['width'], obs['height']
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    
    b = np.zeros((20, 32, 32), dtype=np.float32)
    
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]
        
        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            if unit_id == strs[3]:
                # Position and Cargo
                b[:2, x, y] = (
                    1,
                    (wood + coal + uranium) / 100
                )
            else:
                # Units
                team = int(strs[2])
                cooldown = float(strs[6])
                idx = 2 + (team - obs['player']) % 2 * 3
                b[idx:idx + 3, x, y] = (
                    1,
                    cooldown / 6,
                    (wood + coal + uranium) / 100
                )
        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            idx = 8 + (team - obs['player']) % 2 * 2
            b[idx:idx + 2, x, y] = (
                1,
                cities[city_id]
            )
        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{'wood': 12, 'coal': 13, 'uranium': 14}[r_type], x, y] = amt / 800
        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            b[15 + (team - obs['player']) % 2, :] = min(rp, 200) / 200
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10
    
    # Day/Night Cycle
    b[17, :] = obs['step'] % 40 / 40
    # Turns
    b[18, :] = obs['step'] / 360
    # Map Size
    b[19, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

    return b


if __name__ == "__main__":


# obs = np.zeros((32,20,32,32))
# actions = np.ones((32))
# next_obs = np.zeros((32,20,32,32))
# dones = np.array([True]*32)
# infos = np.array([{}]*32)
# cat_parts = {"obs":obs, "acts":actions, "next_obs":next_obs, "dones":dones, "infos":infos}
# transitions = types.Transitions(**cat_parts)

    seed = 42
    seed_everything(seed)

    episode_dir = '/home/ubuntu/work/codes/imitation_learning/input/episodes'
    obses, samples = create_dataset_from_json(episode_dir)
    print('obses:', len(obses), 'samples:', len(samples))

    labels = [sample[-1] for sample in samples]
    actions = ['north', 'south', 'west', 'east', 'bcity']
    for value, count in zip(*np.unique(labels, return_counts=True)):
        print(f'{actions[value]:^5}: {count:>3}')

    trainsitions = []
    acts = []
    obs = []
    for sample in tqdm(samples[:100]):
        acts.append(sample[2])
        obs.append(make_input(obses[sample[0]], sample[1]))
    #BCはobsとactsしか使わないので他は適当
    # trajectory_length = len(samples)
    trajectory_length = 100
    cat_parts = {"obs":np.stack(obs, axis=0).astype("float64"), "acts":np.stack(acts, axis=0).astype("float64"), "next_obs":np.zeros((trajectory_length,20,32,32)), "dones":np.array([True]*trajectory_length), "infos":np.array([{}]*trajectory_length)}
    transitions = types.Transitions(**cat_parts)

    pass






