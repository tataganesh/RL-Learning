import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import utils
from collections import defaultdict
env = gym.make('CartPole-v0')


class expectedSarsa:
    def __init__(self, config, descretize_config, file_utils):
        self.NUM_EPISODES = config["num_episodes"]
        num_actions = env.action_space.n
        self.actions = range(0, num_actions)
        state_powers = descretize_config["state_powers"]
        num_bins = config["num_bins"]
        low, high = descretize_config["low"], descretize_config["high"]
        self.descretizer = utils.Descretize(num_bins, state_powers, env.observation_space, low, high)
        num_states_approx = int(sum(max(num_bins) ** power for power in state_powers))
        self.Q = np.zeros((num_states_approx, num_actions))
        self.step_size = config["step_size"]
        self.epsilon = config["epsilon"]
        self.random_gen = np.random.RandomState(config["random_seed"])
        self.alpha_decay = config["alpha_decay"]
        self.epsilon_decay = config["epsilon_decay"]
        self.optimal_count = 0
        self.normal_count  = 0
        self.max_time_steps = 500
        self.file_utils = file_utils
        self.data = defaultdict(list)
        env.seed(config["random_seed"])

    def argmax(self, action_values):
        return self.random_gen.choice(np.flatnonzero(action_values == action_values.max()))

    def q_value_update(self, current_state, current_action, next_state, reward, done):
        expected_return = 0
        greedy_action = np.argmax(self.Q[next_state, :])
        expected_return = (1 - self.epsilon) * self.Q[next_state, greedy_action] + self.epsilon * self.Q[next_state, 1 - greedy_action]
        self.Q[current_state, current_action] += self.step_size * (reward + expected_return * (1 - done) - \
                                        self.Q[current_state, current_action])

    
    def policy(self, state):
        if self.random_gen.random() < self.epsilon:
            action = self.random_gen.choice(self.actions)
        else:
            action = self.argmax(self.Q[state, :])
        return action
        
    
    def train(self):
        for i_episode in range(self.NUM_EPISODES):
            observation = env.reset()

            current_state = self.descretizer.get_state_value(observation)
            for t in range(self.max_time_steps):
                # if not i_episode % 100:
                #     env.render()
                action = self.policy(current_state)
                observation, reward, done, info = env.step(action)
                next_state = self.descretizer.get_state_value(observation)
                self.q_value_update(current_state, action, next_state, reward, done)
                current_state = next_state
                if done:
                    episode_reward = t + 1
                    print(f"Episode {i_episode} finished after {t + 1} timesteps with {episode_reward} reward")
                    self.step_size = self.step_size * self.alpha_decay
                    self.epsilon = self.epsilon * self.epsilon_decay
                    self.data["step_size"].append(self.step_size)
                    self.data["epsilon"].append(self.epsilon)
                    self.data['episode_reward'].append(episode_reward)
                    self.data["mean_reward"].append(np.mean(self.data["episode_reward"]))
                    break
            
        print(f"Average reward - {self.data['mean_reward'][-1]}")
        self.file_utils.save(self.data)
        return self.data['episode_reward']
        
def run_agent(config_path):
    run_config = utils.load_config(config_path)
    config = run_config["agent_params"]
    run_params = run_config["run_params"]
    descretize_config = {"low": [-2.4, -4, -0.5,-4], "high": [2.4, 2, 0.5, 2],  "state_powers": [1, 2, 3, 4]}
    file_utils = utils.FileUtils(run_params["save_path"], run_params["name"])
    file_utils.copy(config_path)
    expectedSarsaAgent = expectedSarsa(config, descretize_config, file_utils)
    episode_rewards = expectedSarsaAgent.train()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agent based on config")
    parser.add_argument("--config", help='Path to configuration file specifying run parameters', required=True)
    args = parser.parse_args()
    run_agent(args.config)