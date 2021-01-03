from cartpole_tilec import run_agent
from utils import FileUtils
import itertools


sweep_config = { 
    "params": {
        "step_size": [0.1], 
        "epsilon": [0.1],
        "random_seed": [200, 300, 3257, 4333, 2366, 1458, 1385, 268, 2705, 1263, 26],
        "alpha_decay": [0.997],
        "epsilon_decay": [0.999],
        "num_episodes": [1000],
        "td_update_algo": ["qlearning"]
    },
    "save_params":{
        "save_path": "sweeps",
        "name": "random_seed_sweep_qlearning_tilec"
    }
}

extra_keys = {
    "tile_coding_params": {
        "low": [
            -3.4,
            -7,
            -1,
            -4
        ],
        "high": [
            3.4,
            7,
            1,
            4
        ],
        "num_tilings": 16,
        "num_tiles": 8,
        "iht": 4096
    }
}

sweep_config.update(extra_keys)
save_params = sweep_config["save_params"]
sweep_params = sweep_config["params"]
fileutils = FileUtils(save_params["save_path"], save_params["name"])
fileutils.save_json(sweep_config, "sweep_config.json")
keys, values = sweep_params.keys(), sweep_params.values()

sweep_combs = [dict(zip(keys, v)) for v in itertools.product(*values)]

for i, conf in enumerate(sweep_combs):
    run_config = dict()
    run_config["agent_params"] = conf
    run_config["save_params"] = {"save_path": fileutils.run_path, "name": str(i)}
    for key, value in extra_keys.items():
        run_config[key] = value
    run_agent(run_config, test_agent=False)
    print(f"\n######   Completed run {i}   ########\n")    

