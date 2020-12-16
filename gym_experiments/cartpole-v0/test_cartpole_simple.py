import pytest
import numpy as np
import gym

from cartpole_qlearning import descretize, qLearning
env = gym.make('CartPole-v0')

class TestCartpoleWithDescrete:
    descretize_config = {"low": [-2.4, -4, -0.5,-4], "high": [2.4, 2, 0.5, 2],  "state_powers": [1, 2, 3, 4]}
    config = {"num_bins": [10, 10, 10, 10], "step_size": 1, "epsilon": 0.1, "random_seed": 66, "alpha_decay": 0.995, "epsilon_decay": 0.999, "num_episodes": 10}
    test_state = 43

    def test_descritization(self):
        observation = np.array([-0.02662141, -0.00072983, -0.00317916, 0.0273904])

        descretizer = descretize(self.config["num_bins"],  self.descretize_config["state_powers"], env.observation_space)
        expected_state_value = 780
        assert descretizer.get_state_value(observation) == expected_state_value

    def test_policy(self):
        test_agent = qLearning(self.config, self.descretize_config)
        action = test_agent.policy(self.test_state)
        assert action == 1




