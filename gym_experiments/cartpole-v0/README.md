## Carpole-V0 Experiments

# Q-Learning

* [Wandb report](https://wandb.ai/tataganesh/RL-Learning-gym_experiments_cartpole-v0/reports/Analysis-of-Average-Reward-across-different-runs--VmlldzozNjgzODI) on the performance of the Q-Learning agent on 3 different step sizes across 10 different random seeds. The trials were divided into two groups of 5 random seeds for each step size. 

* Tests - 

   `pytest test_cartpole_simple.py::TestCartpoleDescreteQLearning` 



# Expected SARSA

* Running Expected SARSA tests - 

   `pytest test_cartpole_simple.py::TestCartpoleDescreteExpectedSarsa`



# Linear Function Approximation with Tile Coding

* Q-Learning

    ![Q-Learning test gif](recordings/cartpole_qlearning.gif)

* SARSA

    ![SARSA test gif](recordings/cartpole_sarsa.gif)




# Sweep Example

* Run sweep with config (TODO: Pass config as command-line argument)

   ` python3 sweep.py `
* Visualize results of a sweep

   ` python3 plotting.py --sweep_path=/home/tata/Learn/RL/RL-Learning/gym_experiments/cartpole-v0/sweeps/random_seed_sweep_qlearning_tilec `