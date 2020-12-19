import matplotlib.pyplot as plt
import utils
import argparse
import numpy as np

# parser = argparse.ArgumentParser(description="Plotting utility")
# parser.add_argument("--run_path", help='Path to configuration file specifying run parameters', required=True)
# args = parser.parse_args()

run_path = '/home/tata/Learn/RL/RL-Learning/gym_experiments/cartpole-v0/expected_sarsa_runs/test_run'
file_utils = utils.plottingUtils(run_path)
data = file_utils.load()
data["episode"] = np.arange(0, data["mean_reward"].shape[0])
y_axes = ['mean_reward', 'episode_reward']
x_axes = ['step_size', 'epsilon', 'episode']

for x_axis in x_axes:
    for y_axis in y_axes:
        plt.xlabel(x_axis, fontsize=18)
        plt.ylabel(y_axis, fontsize=16)
        plt.plot(data[x_axis], data[y_axis])
        plt.show()