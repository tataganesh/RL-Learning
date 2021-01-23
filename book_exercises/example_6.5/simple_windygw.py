import numpy as np
import gym
import gym_gridworlds
import matplotlib.pyplot as plt

env = gym.make('WindyGridworld-v0')  # substitute environment's name
env.reset()

# print(env.actions)

class SARSA:
    def __init__(self):
        self.Q = np.zeros((70, env.action_space.n))
        self.alpha = 0.5
        self.epsilon = 0.1
        self.random_gen = np.random.RandomState(10)
        self.actions = np.arange(0, env.action_space.n)


    def argmax(self, action_values):
        return self.random_gen.choice(np.flatnonzero(action_values == action_values.max()))
    
    def state_to_number(self, state_cdnts):
        return state_cdnts[0] * 10 + state_cdnts[1]
    
    def policy(self, state):
        if self.random_gen.random() < self.epsilon:
            action = self.random_gen.choice(self.actions)
        else:
            action = self.argmax(self.Q[state, :])
        return action

    def update(self, s, a, r, s_next, a_next):
        self.Q[s,a] += self.alpha * (r + self.Q[s_next, a_next] - self.Q[s, a])
        
    def train(self):
        time_steps = list()
        total_time_steps = 0
        for i_episode in range(200):
            observation = env.reset()
            state = self.state_to_number(observation)
            # observation, reward, done, info = env.step(cur_action)
            # env.render()
            cur_action = self.policy(state)
            for t in range(10000):
                observation, reward, done, info = env.step(cur_action) # take a random action
                next_state = self.state_to_number(observation)
                next_action = self.policy(next_state)
                self.update(state, cur_action, reward,  next_state, next_action)
                state = next_state
                cur_action = next_action
                time_steps.append((i_episode, total_time_steps))
                total_time_steps += 1
                if done:
                    print(f"Episode - {i_episode}, Time steps taken - {t}")
                    # time_steps.append(t)
                    break
        episodes, time_steps = list(zip(*time_steps))
        plt.plot(time_steps, episodes)
        # plt.ylim(0, 100)
        plt.show()


agent = SARSA()
agent.train()