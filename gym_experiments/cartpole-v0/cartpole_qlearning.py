import gym
import numpy as np
import matplotlib.pyplot as plt
# import wandb
from utils import Descretize

env = gym.make('CartPole-v0')


class qLearning:
    def __init__(self, config, descretize_config):
        
        self.NUM_EPISODES = config["num_episodes"]
        num_actions = env.action_space.n
        self.actions = range(0, num_actions)
        state_powers = descretize_config["state_powers"]
        num_bins = config["num_bins"]
        low, high = descretize_config["low"], descretize_config["high"]
        self.descretizer = Descretize(num_bins, state_powers, env.observation_space, low, high)
        
        num_states_approx = int(sum(max(num_bins) ** power for power in state_powers))
        self.Q = np.zeros((num_states_approx, num_actions))
        self.step_size = config["step_size"]
        self.epsilon = config["epsilon"]
        self.random_gen = np.random.RandomState(config["random_seed"])
        env.seed(config["random_seed"])
        self.alpha_decay = config["alpha_decay"]
        self.epsilon_decay = config["epsilon_decay"]
        self.optimal_count = 0
        self.normal_count  = 0
        self.max_time_steps = 500

    def argmax(self, action_values):
        return self.random_gen.choice(np.flatnonzero(action_values == action_values.max()))

    def q_value_update(self, current_state, current_action, next_state, reward, done):
        self.Q[current_state, current_action] += self.step_size * (reward + np.max(self.Q[next_state, :]) * (1 - done) - \
                                        self.Q[current_state, current_action])

    
    def policy(self, state):
        action = None
        if self.random_gen.random() < self.epsilon:
            action = self.random_gen.choice(self.actions)
        else:
            action = self.argmax(self.Q[state, :])
        return action
        
    
    def train(self):
        reward_history = list()
        for i_episode in range(self.NUM_EPISODES):
            observation = env.reset()

            current_state = self.descretizer.get_state_value(observation)
            for t in range(self.max_time_steps):
                # env.render()
                action = self.policy(current_state)
                observation, reward, done, info = env.step(action)
                next_state = self.descretizer.get_state_value(observation)
                self.q_value_update(current_state, action, next_state, reward, done)
                current_state = next_state
                if done:
                    print(f"Episode {i_episode} finished after {t + 1} timesteps with {t + 1} reward")
                    # wandb.log({"epsilon": self.epsilon})
                    self.step_size = self.step_size * self.alpha_decay
                    self.epsilon = self.epsilon * self.epsilon_decay
                    reward_history.append(t + 1)
                    break
            # wandb.log({"mean_reward_50": np.mean(reward_history[-50:]), "mean_reward": np.mean(reward_history), "Episode reward": reward_history[-1], "episode": i_episode})
            
        print(f"Average reward - {np.mean(reward_history)}")
        return reward_history
        

config = {"num_bins": [10, 10, 10, 10], "step_size": 0.1, "epsilon": 0.3, "random_seed": 100, "alpha_decay": 0.997, "epsilon_decay": 1, "num_episodes": 1000}

def main():
    # wandb.init(project="CartPole-v0_Q-Learning_State-Discretization", config = config)
    descretize_config = {"low": [-2.4, -4, -0.5,-4], "high": [2.4, 2, 0.5, 2],  "state_powers": [1, 2, 3, 4]}
    qLearningAgent = qLearning(config,descretize_config)
    episode_rewards = qLearningAgent.train()
    env.close()


if __name__ == "__main__":
    main()