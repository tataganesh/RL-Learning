import matplotlib.pyplot as plt
import utils
import argparse
import numpy as np
from pathlib import Path
import math

# parser = argparse.ArgumentParser(description="Plotting utility")
# parser.add_argument("--run_path", help='Path to configuration file specifying run parameters', required=True)
# args = parser.parse_args()


y_axes = ['mean_reward', 'episode_reward']
x_axes = ['step_size', 'epsilon', 'episode']
def plot_run(run_path, save=False):
    print(f"Plotting {run_path}")
    EPISODE_KEY = "episode"
    plotting_utils = utils.plottingUtils(run_path)
    data = plotting_utils.load()
    if EPISODE_KEY not in data:
        data["episode"] = np.arange(0, data[x_axes[0]].shape[0])
    for x_axis in x_axes:
        for y_axis in y_axes:
            plt.xlabel(x_axis, fontsize=18)
            plt.ylabel(y_axis, fontsize=16)
            plt.plot(data[x_axis], data[y_axis])
            # plt.show()
            if save:
                plotting_utils.save(x_axis, y_axis, plt)
            plt.clf()
    return data["mean_reward"]

def plot_range_splits(values, num_splits):
    colors = ['green', 'blue', 'yellow']
    x = range(values.shape[1])
    n = math.ceil(values.shape[0] / num_splits)
    for i in range(0, num_splits):
        print(values[i*n: i*n + n].shape)
        max_values = np.max(values[i*n: i*n + n], axis=0)
        min_values = np.min(values[i*n: i*n + n], axis=0)
        avg_values = np.mean(values[i*n: i*n + n], axis=0)
        plt.fill_between(x, max_values, min_values, color=colors[i])
        plt.plot(x, avg_values, color='red')




def plot_sweep(sweep_path, save=False):
    mean_rewards = list()
    plotting_utils = utils.plottingUtils(sweep_path)
    for p in Path(sweep_path).iterdir():
        if not p.is_dir() or p.name == "plots":
            continue
        run_path = str(p)
        rewards = plot_run(run_path, save)
        print(rewards.shape)
        mean_rewards.append(rewards)

    stacked_rewards = np.vstack([r for r in mean_rewards])
    max_values = np.max(stacked_rewards, axis=0)
    min_values = np.min(stacked_rewards, axis=0)
    x = np.arange(0, min_values.shape[0])
    plot_range_splits(stacked_rewards, 2)
    # plt.fill_between(x, max_values, min_values)
    
    plt.title("Average reward for two-different groups of random seeds")
    # plt.show()
    if save:
        plotting_utils.save("Episodes", "Average Reward", plt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run / Sweep Plotting")
    parser.add_argument("--run_path", help='Path to run folder')
    parser.add_argument("--sweep_path", help="Path to sweep folder")
    parser.add_argument("--save", help="Save plots in 'plots' folder inside the run folder",  action='store_true')
    args = parser.parse_args()
    if args.run_path and args.sweep_path:
        print("Cannot plot both a run and sweep. Please choose only one of them")
    elif args.run_path:
        plot_run(args.run_path, args.save)
    elif args.sweep_path:
        plot_sweep(args.sweep_path, args.save)
