import os
import argparse
from collections import defaultdict
import sys
sys.path.append("../../helpers")

import gym
import numpy as np
import matplotlib.pyplot as plt
from log_setup import get_logger

import utils
from SGD import sgd_factory

env = gym.make('CartPole-v0')

class CartPoleWithTileC:
    SARSA = "sarsa"
    ESARSA = "expected_sarsa"
    QLEARNING = "qlearning"
    def __init__(self, config, tile_coding_config, file_utils):
        self.NUM_EPISODES = config["num_episodes"]
        num_actions = env.action_space.n
        self.actions = range(0, num_actions)
        low, high = tile_coding_config["low"], tile_coding_config["high"]
        num_tilings, num_tiles, iht_size = tile_coding_config["num_tilings"], tile_coding_config["num_tiles"], tile_coding_config["iht"]
        self.step_size = config["step_size"]
        self.epsilon = config["epsilon"]
        self.random_gen = np.random.RandomState(config["random_seed"])
        self.alpha_decay = config["alpha_decay"]
        self.epsilon_decay = config["epsilon_decay"]
        self.max_time_steps = 500
        self.file_utils = file_utils
        self.data = defaultdict(list)
        env.seed(config["random_seed"])
        self.tile_coder = utils.TileCoderFeature(low, high, num_tilings, num_tiles, iht_size)
        sgd_reg = sgd_factory("sklearn")
        self.model = sgd_reg(self.step_size, num_actions, iht_size, random_state=config["random_seed"])
        self.feature_vector = np.zeros(iht_size)
        self.td_update_algo = config["td_update_algo"]
        self.plan = False
        if "planning_config" in config:
            planning_config = config["planning_config"]
            self.plan = True
            self.planner = utils.Replay(planning_config["buffer_size"], planning_config["sample_size"], planning_config["steps"], self.random_gen)
        self.logger = get_logger(self.file_utils.run_name, os.path.join(self.file_utils.run_path, "cartpole_tile.log"))

    def argmax(self, action_values):
        return self.random_gen.choice(np.flatnonzero(action_values == action_values.max()))

    def get_td_update_value(self, tiles, action):
        if self.td_update_algo == self.SARSA:
            return self.get_action_value(tiles, action)
        else:
            greedy_action = np.argmax(np.array([self.get_action_value(tiles, a) for a in self.actions]))
            if self.td_update_algo == self.ESARSA:
                td_update_val = (1 - self.epsilon) * self.get_action_value(tiles, greedy_action) + self.epsilon * self.get_action_value(tiles, 1 - greedy_action)
                return td_update_val
            elif self.td_update_algo == self.QLEARNING:
                return self.get_action_value(tiles, greedy_action)

    def get_action_value(self, tiles, action):
        action_value = self.model.run(tiles, action) 
        return action_value
    
    def policy(self, tiles):
        if self.random_gen.random() < self.epsilon:
            action = self.random_gen.choice(self.actions)
        else:
            action_values = np.array([self.get_action_value(tiles, a) for a in self.actions])
            action = self.argmax(action_values)
        return action
        
    def train(self):
        for i_episode in range(self.NUM_EPISODES):
            observation = env.reset()
            cur_tiles = self.tile_coder.get_tiles(observation) 
            cur_action = self.policy(cur_tiles)
            cur_state_action_val = self.get_action_value(cur_tiles, cur_action)  
            for t in range(self.max_time_steps):
                # if not i_episode % 100:
                #   env.render()
                observation, reward, done, info = env.step(cur_action)
                next_tiles = self.tile_coder.get_tiles(observation)
                next_action = self.policy(next_tiles)
                next_state_action_val = self.get_td_update_value(next_tiles, next_action)
                self.model.update(cur_tiles, [reward + next_state_action_val * (1 - done)], cur_action)
                if self.plan:
                    self.planner.append(cur_tiles, cur_action, next_tiles, reward, done)
                    self.model = self.planner.planning(self, self.model.copy())
                cur_action = next_action
                cur_state_action_val = next_state_action_val
                cur_tiles = next_tiles
                if done:
                    episode_reward = t + 1
                    self.data["step_size"].append(self.step_size)
                    self.data["epsilon"].append(self.epsilon)
                    self.data['episode_reward'].append(episode_reward)
                    self.data["mean_reward"].append(np.mean(self.data["episode_reward"]))
                    if not i_episode % 10:
                        self.logger.info(f"Episode {i_episode} finished after {t + 1} timesteps with {episode_reward} reward, epsilon {self.epsilon}, mean reward {self.data['mean_reward'][-1]}")
                    self.step_size = self.step_size * self.alpha_decay
                    self.epsilon = self.epsilon * self.epsilon_decay
                    break
            
        self.logger.info(f"Average reward - {self.data['mean_reward'][-1]}")
        self.file_utils.save(self.data)
        self.model.save(os.path.join(self.file_utils.run_path, "regressors"))
        return self.data['episode_reward']

    def test(self, folder_path):
        global env
        env = gym.wrappers.Monitor(env, os.path.join(self.file_utils.run_path, 'cartpole_tilec_recording'), force=True)
        self.model.load(folder_path)
        self.epsilon = 0 # No Exploration            
        observation = env.reset()
        cur_action = self.policy(self.tile_coder.get_tiles(observation))
        for t in range(self.max_time_steps):
            env.render()
            observation, reward, done, info = env.step(cur_action)
            cur_action = self.policy(self.tile_coder.get_tiles(observation))
            if done:
                print(f"Testing done. Total Reward - {t + 1}")
                break

        
def run_agent(run_config, config_path = None, test_agent = True):
    config = run_config["agent_params"]
    save_params = run_config["save_params"]
    tile_coding_config = run_config["tile_coding_params"]
    file_utils = utils.FileUtils(save_params["save_path"], save_params["name"])
    if config_path:
        file_utils.copy(config_path)
    else:
        file_utils.save_json(run_config, "config.json")
    cartpoleAgent = CartPoleWithTileC(config, tile_coding_config, file_utils)
    episode_rewards = cartpoleAgent.train()
    if test_agent:
        cartpoleAgent.test(os.path.join(file_utils.run_path, 'regressors'))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agent based on config")
    parser.add_argument("--config", help='Path to configuration file specifying run parameters', required=True)
    args = parser.parse_args()
    run_config = utils.load_config(args.config)
    run_agent(run_config, args.config)