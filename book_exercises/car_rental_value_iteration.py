import math
from functools import lru_cache
import random
import time
import numpy as np
from pprint import pprint


MAX_MOVE = 5
MAX_CARS = 20
MOVE_COST = -2

REQUESTS_1 = 3
REQUESTS_2 = 4

RETURNS_1 = 3
RETURNS_2 = 2

UPPER_LIMIT = 9

@lru_cache(maxsize=64)
def poisson(x, lmbda):
    return math.pow(lmbda, x) * math.exp(-1 * lmbda) / math.factorial(x)



class DPagent:
    def __init__(self, gamma):
        self.gamma = gamma
        self.stateValues = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
        self.policy = np.zeros_like(self.stateValues)
        self.actions = np.arange(-MAX_MOVE, MAX_MOVE + 1)
        self.delta = 0.1


    def bellman(self, action, state):
        expected_return = 0
        expected_return += MOVE_COST * abs(action)
        all_v = list()
        all_values = list()
        for req_loc1 in range(0, UPPER_LIMIT):
            num_cars_first_location = int(max(state[0] - action, 0))
            possible_rental_first_loc = min(num_cars_first_location, req_loc1)
            num_cars_first_location -= possible_rental_first_loc
            for req_loc2 in range(0, UPPER_LIMIT):

                num_cars_second_location = int(min(state[1] + action, MAX_CARS))
                possible_rental_second_loc = min(num_cars_second_location, req_loc2)
                num_cars_second_location -= possible_rental_second_loc
                reward = (possible_rental_first_loc + possible_rental_second_loc) * 10
                prob_req1_req2 = poisson(req_loc1, REQUESTS_1) * poisson(req_loc2, REQUESTS_2)
                for ret_loc1 in range(0, UPPER_LIMIT):
                    all_v.append([])
                    num_cars_first_location_ = min(num_cars_first_location + ret_loc1, MAX_CARS)
                    for ret_loc2 in range(0, UPPER_LIMIT):
                        num_cars_second_location_ = min(num_cars_second_location + ret_loc2, MAX_CARS)
                        total_prob = poisson(ret_loc1, RETURNS_1) * poisson(ret_loc2, RETURNS_2) * prob_req1_req2
                        # Bellman update
                        expected_return += total_prob * (reward + self.gamma * self.stateValues[num_cars_first_location_, num_cars_second_location_])
                        # all_probs.append((poisson(req_loc1, REQUESTS_1), poisson(req_loc2, REQUESTS_2), poisson(ret_loc1, RETURNS_1) , poisson(ret_loc2, RETURNS_2)))
                        all_values.append((reward, total_prob, [num_cars_first_location_, num_cars_second_location_]))
                        all_v[-1].append(reward + self.gamma * self.stateValues[num_cars_first_location_, num_cars_second_location_])
        return expected_return 


    def value_iteration(self):
        while True:
            old_V = np.copy(self.stateValues)
            new_V = np.zeros_like(self.stateValues)
            for loc1 in range(0, MAX_CARS + 1):
                for loc2 in range(0, MAX_CARS + 1):
                    print(loc1, loc2)
                    # print(loc1, loc2)
                    state = loc1, loc2
                    max_value = -1
                    greedy_action = self.actions[0]
                    for action in self.actions:
                        if not( (action >= 0 and state[0] >= action) or (action < 0 and state[1] >= abs(action))):
                            action_value = -float('inf')
                        else:
                            action_value = self.bellman(action, state)
                        if max_value < action_value:
                            max_value = action_value
                            greedy_action = action
                    new_V[state[0], state[1]] = max_value
                    self.policy[state[0], state[1]] = greedy_action
            self.stateValues = new_V
            error = np.abs(old_V - self.stateValues).sum()
            # print(self.stateValues)
            print(error)
            if error < self.delta:
                break

        np.save(f"value_it_policy.npy", self.policy)
        # np.save(f"value_it_values_{POLICY_NUM}.npy", self.stateValues)


dp = DPagent(gamma=0.9)
dp.value_iteration()