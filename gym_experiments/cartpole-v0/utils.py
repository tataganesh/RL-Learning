import numpy as np
import os
import shutil
import json
import errno
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import tilecoder
import random
class Descretize:
    def __init__(self, num_bins, state_powers, observation_space, low=None, high=None):
        # low = observation_space.low
        # high = observation_space.high
        low = observation_space.low if low is None else low
        high = observation_space.high if high is None else high
        self.bins = [np.linspace(l, h, num_bins[i]) for i, (l, h) in enumerate(zip(low, high))]
        self.state_powers = state_powers
    
    def get_state_value(self, observation):
        state_bins = [np.digitize(observation[i], np.array(self.bins[i])) for i in range(len(self.bins))]
        state_value = int(sum(state_bin ** power for state_bin, power in zip(state_bins, self.state_powers)))
        return state_value

    

def load_config(config_path):
    with open(config_path, 'r') as fp:
        config = json.load(fp)
    return config


class FileUtils:

    def __init__(self, folder_path, run_name):
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        run_path = os.path.join(folder_path, run_name)
        if not os.path.exists(run_path):
            os.mkdir(run_path)
        self.folder_path = folder_path
        self.run_path = run_path
    
    def save(self, data):
        for data_name, data_value in data.items():
            data_value = np.array(data_value)
            data_file_path = os.path.join(self.run_path, f"{data_name}.npy")
            np.save(data_file_path, data_value)
            print(f"Saved {data_name} in {data_file_path} successfully")

    def copy(self, file_path):
        file_name = file_path.split("/")[-1]
        dest_path = os.path.join(self.run_path, file_name)
        shutil.copyfile(file_path, dest_path)

    def save_json(self, obj, name):
        file_path = os.path.join(self.run_path, name)
        with open(file_path, 'w') as fp:
            json.dump(obj, fp)

class plottingUtils:
    NUMPY_EXTENSION = "*.npy"
    PLOTS_DIR = "plots"
    def __init__(self, run_path):
        if not os.path.exists(run_path):
            raise Exception(f"{run_path} does not exist")
        self.run_path = run_path
        self.plots_path = os.path.join(self.run_path, self.PLOTS_DIR)
        if not (os.path.exists(self.plots_path) and os.path.isdir(self.plots_path)):
            os.mkdir(self.plots_path)    
    def load(self):
        data = dict()
        for file_path in Path(self.run_path).glob(self.NUMPY_EXTENSION):
            file_name = file_path.stem
            data_value = np.load(str(file_path))
            data[file_name] = data_value
        return data

    def save(self, x, y, plt):
        plot_name = f"{y} vs {x}"
        plot_file_path = os.path.join(self.plots_path, plot_name)
        plt.savefig(plot_file_path)


class TileCoderFeature:
    
    def __init__(self, low, high, num_tilings, num_tiles, iht_size):
        self.low = low
        self.high = high
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.iht = tilecoder.IHT(iht_size)

    def get_tiles(self, inp):
        scaled_values = [((inp[i] - self.low[i]) / (self.high[i] - self.low[i])) * self.num_tiles for i in range(len(inp))]
        # scaled_values = [(inp[i] / (self.high[i] - self.low[i])) * self.num_tiles for i in range(len(inp))]
        return tilecoder.tiles(self.iht, self.num_tilings, scaled_values)

class Replay:
    # Currently only for expected SARSA and Q-Learning

    def __init__(self, buffer_size, sample_size, planning_steps, random_gen):
        
        self.buffer_size = buffer_size
        self.sample_size = sample_size
        self.planning_steps = planning_steps
        self.buffer = []
        self.random_gen = random_gen

    def planning(self, agent, learning_model):
        
        for step in range(self.planning_steps):
            experience_idx = np.unique(self.random_gen.randint(low=0, high=len(self.buffer), size=self.sample_size))
            for i in experience_idx:
                state, action, next_state, reward, is_terminal = self.buffer[i]
                next_state_action_val = agent.get_td_update_value(next_state, None) # Planning only for expected SARSA and Q-Learning
                learning_model.update(state, [reward + next_state_action_val * (1 - is_terminal)], action)
        return learning_model

    def append(self, state, action, next_state, reward, is_terminal):
        experience = [state, action, next_state, reward, is_terminal]
        if len(self.buffer) == self.buffer_size:
            self.buffer[-1] = experience
        else:
            self.buffer.append(experience)


    

    




            


