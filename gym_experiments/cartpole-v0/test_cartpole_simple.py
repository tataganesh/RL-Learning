import math
import shutil

import pytest
import numpy as np
import gym

from cartpole_qlearning import qLearning
from cartpole_expectedSarsa import expectedSarsa
import utils
from SGD import sgd_factory

env = gym.make('CartPole-v0')

class TestCartpoleDescreteQLearning:
    descretize_config = {"low": [-2.4, -4, -0.5,-4], "high": [2.4, 2, 0.5, 2],  "state_powers": [1, 2, 3, 4]}
    config = {"num_bins": [10, 10, 10, 10], "step_size": 1, "epsilon": 0.1, "random_seed": 66, "alpha_decay": 0.995, "epsilon_decay": 0.999, "num_episodes": 20}
    test_state = 43

    def test_descritization(self):
        observation = np.array([-0.02662141, -0.00072983, -0.00317916, 0.0273904])

        descretizer = utils.Descretize(self.config["num_bins"],  self.descretize_config["state_powers"], env.observation_space)
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
    file_utils = utils.FileUtils(test_folder, "test")
    def test_descritization(self):
        observation = np.array([-0.02662141, -0.00072983, -0.00317916, 0.0273904])

        descretizer = utils.Descretize(self.config["num_bins"],  self.descretize_config["state_powers"], env.observation_space)
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

class TestCartpoleTileCoder:
    low = [-2.4, -4, -0.5,-4]
    high = [2.4, 2, 0.5, 2]
    num_tilings = 10
    num_tiles = 10
    iht_size = 3000
    iht_sgd = 16
    test_ob1 = [-0.02662141, -0.00072983, -0.00317916, 0.0273904]
    expected_tiles_ob1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    test_ob2 = [-0.21297128, -0.00583864, -0.02543328, 0.2191232]
    expected_tiles_ob2 = [10, 11, 12, 13, 14, 5, 15, 7, 8, 9]
    learning_rate = 0.01
    num_actions = 2

    def test_tile_coder(self):
        tile_coder = utils.TileCoderFeature(self.low, self.high, self.num_tilings, self.num_tiles, self.iht_size)
        tiles = tile_coder.get_tiles(self.test_ob1)
        assert tiles == self.expected_tiles_ob1, "First observation Tile Coder encoding failed"
        tiles = tile_coder.get_tiles(self.test_ob2)
        assert tiles == self.expected_tiles_ob2, "Second observation Tile Coder encoding failed"
    
    def test_sgd_update(self):

        expected_weights = np.array([[0, 0, 0, 0, 0, 0.003, 0, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003],
                                        [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0, 0,  0, 0, 0, 0]])
        sgd = sgd_factory("sklearn")
        model = sgd(self.learning_rate, self.num_actions, self.iht_sgd)
        model.update(self.expected_tiles_ob1, [0.2], 1)
        model.update(self.expected_tiles_ob2, [0.3], 0)
        assert np.allclose(model.weights, expected_weights)
        
    def test_sgd_run(self):
        expected_value = 0.01
        sgd = sgd_factory("sklearn")
        model = sgd(self.learning_rate, self.num_actions, self.iht_sgd)
        model.update(self.expected_tiles_ob1, [0.2], 1)
        assert math.isclose(model.run(self.expected_tiles_ob2, 1), expected_value)


    def test_planning(self):
        pass












