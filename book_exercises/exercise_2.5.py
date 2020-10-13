import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
NUM_RUNS = 200
NUM_STEPS = 15000
NUM_ACTIONS = 10

def argmax(q_values):
    all_max_values = list()
    current_max = -1
    for i, val in enumerate(q_values):
        if val > current_max:
            current_max = val
            all_max_values = [i]
        elif val == current_max:
            all_max_values.append(i)
    return np.random.choice(all_max_values)


class epsilon_greedy_agent:
    def __init__(self, distribution_shift_steps = NUM_STEPS/2, alpha = None, epsilon=0.1):
        # np.random.seed(123)
        self.q_original_values = [np.random.normal() for action in range(NUM_ACTIONS)]
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_values = [0] * NUM_ACTIONS
        self.action_counts = [0] * NUM_ACTIONS
        self.last_action = argmax(self.q_values)
        self.distribution_shift_steps = distribution_shift_steps
        self.num_steps = 0
        self.max_action_value = max(self.q_original_values)
        self.optimal_action = argmax(self.q_original_values)
    
    def generate_reward(self):
        return np.random.normal(loc = self.q_original_values[self.last_action])
    
    def step(self):
        self.num_steps += 1
        if self.num_steps  > self.distribution_shift_steps:
            self.q_original_values = [qval + np.random.normal(scale=0.01) for qval in self.q_original_values]
        self.max_action_value = max(self.q_original_values)
        self.optimal_action = argmax(self.q_original_values)

        reward = self.generate_reward()
        self.action_counts[self.last_action] += 1
        alpha = self.alpha
        if alpha is None:
            alpha = 1/self.action_counts[self.last_action]
        self.q_values[self.last_action] += alpha * (reward - self.q_values[self.last_action])
        randomP = np.random.random()
        if randomP < self.epsilon:
            current_action = np.random.randint(len(self.q_values))
        else:
            current_action = argmax(self.q_values)
        self.last_action = current_action
        return reward, current_action

#  epsilon_greedy_agent(alpha=0.1
colors = ['green', 'blue', 'red']
max_q_values = list()
all_max_q_values = list()
plots = list()
for i in range(0, 2):
    all_averages = list()
    for run in tqdm(range(NUM_RUNS)):
        if i == 0:
            agent = epsilon_greedy_agent()
        if i == 1:
            agent = epsilon_greedy_agent(alpha=0.1)
        reward_averages = list()
        max_q_values = list()
        rew = list()
        reward_sums = [0]
        np.random.seed(run)
        # print(f"----------RUN {run}-------------")
        for step in range(0, NUM_STEPS):
            max_q_values.append(agent.max_action_value)
            reward, action = agent.step()
            rew.append(reward)
            reward_sums.append(reward_sums[-1] + reward)
            reward_averages.append(reward_sums[-1] / (step + 1) )
        all_averages.append(reward_averages)
        all_max_q_values.append(max_q_values)
    # print( np.mean(all_averages, axis=0))
    res, = plt.plot(range(0, NUM_STEPS), np.mean(all_averages, axis=0), color=colors[i])
    plots.append(res)
    res1, = plt.plot(range(0, NUM_STEPS), np.mean(all_max_q_values, axis=0), color='red')
    plots.append(res1)
plt.legend(handles=plots, labels=['Alpha 1/n','Max action value1', 'Alpha 0.1', 'Max action value2'])
plt.title("Non-Stationary Problem Analysis")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.show()

