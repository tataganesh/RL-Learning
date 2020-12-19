import pytest
import numpy as np
import gym

from cartpole_qlearning import qLearning
from cartpole_expectedSarsa import expectedSarsa
from utils import Descretize, FileUtils
import shutil
env = gym.make('CartPole-v0')

class TestCartpoleDescreteQLearning:
    descretize_config = {"low": [-2.4, -4, -0.5,-4], "high": [2.4, 2, 0.5, 2],  "state_powers": [1, 2, 3, 4]}
    config = {"num_bins": [10, 10, 10, 10], "step_size": 1, "epsilon": 0.1, "random_seed": 66, "alpha_decay": 0.995, "epsilon_decay": 0.999, "num_episodes": 20}
    test_state = 43

    def test_descritization(self):
        observation = np.array([-0.02662141, -0.00072983, -0.00317916, 0.0273904])

        descretizer = Descretize(self.config["num_bins"],  self.descretize_config["state_powers"], env.observation_space)
        expected_state_value = 780
        assert descretizer.get_state_value(observation) == expected_state_value

    def test_policy(self):
        test_agent = qLearning(self.config, self.descretize_config)
        action = test_agent.policy(self.test_state)
        assert action == 1

    def test_training(self):
        test_agent = qLearning(self.config, self.descretize_config)
        episode_rewards = test_agent.train()
        print(episode_rewards)
        assert episode_rewards == [11, 11, 11, 13, 10, 25, 32, 12, 14, 34, 18, 12, 30, 9, 14, 13, 16, 18, 36, 17]





class TestCartpoleDescreteExpectedSarsa:
    descretize_config = {"low": [-2.4, -4, -0.5,-4], "high": [2.4, 2, 0.5, 2],  "state_powers": [1, 2, 3, 4]}
    config = {"num_bins": [10, 10, 10, 10], "step_size": 1, "epsilon": 0.1, "random_seed": 66, "alpha_decay": 0.995, "epsilon_decay": 0.999, "num_episodes": 20}
    test_state = 43
    test_folder = "pytest_esarsa"
    file_utils = FileUtils(test_folder, "test")
    def test_descritization(self):
        observation = np.array([-0.02662141, -0.00072983, -0.00317916, 0.0273904])

        descretizer = Descretize(self.config["num_bins"],  self.descretize_config["state_powers"], env.observation_space)
        expected_state_value = 780
        assert descretizer.get_state_value(observation) == expected_state_value

    def test_policy(self):
        test_agent = expectedSarsa(self.config, self.descretize_config, self.file_utils)
        action = test_agent.policy(self.test_state)
        assert action == 1

    def test_training(self):
        test_agent = expectedSarsa(self.config, self.descretize_config, self.file_utils)
        episode_rewards = test_agent.train()
        assert episode_rewards == [11, 11, 11, 13, 10, 25, 32, 12, 14, 34, 18, 12, 30, 9, 14, 13, 32, 17, 12, 12]

    

    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self, request):
        """Cleanup testing directory once we are finished."""
        def remove_test_dir():
            shutil.rmtree(self.test_folder)
        request.addfinalizer(remove_test_dir)








