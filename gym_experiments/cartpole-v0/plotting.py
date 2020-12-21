import matplotlib.pyplot as plt
import utils
import argparse
import numpy as np

# parser = argparse.ArgumentParser(description="Plotting utility")
# parser.add_argument("--run_path", help='Path to configuration file specifying run parameters', required=True)
# args = parser.parse_args()


y_axes = ['mean_reward', 'episode_reward']
x_axes = ['step_size', 'epsilon', 'episode']

def plot_run(run_path, save=False):
    print(f"Plotting {run_path}")
    print(save)
    EPISODE_KEY = "episode"
    plotting_utils = utils.plottingUtils(run_path, save)
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
        pass
