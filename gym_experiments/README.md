## Performance Evaluation in RL - Some notes [ In Progress ]

Resources

* [Evaluating the Performance of Reinforcement Learning Algorithms](https://arxiv.org/pdf/2006.16958.pdf)






# Experimation Workflow(s)

## Custom

* **Logging** - Use logger object to print stdout. This will help to keep track of each run as separate log files. Errors / Exceptions can also be logged in the same file. 
* **Runs** - Generate unique run name folder. It can be passed as a command-line argument. Or, we could have a run_config.json in which the run name is specified. This config.json is copied to the run folder.
* **Sweep** - Generate unique sweep name folder. Specify sweep characteristics using a sweep_config.json. 


## Misc. Notes

* Always, **ALWAYS**, observe the impact of changing random seeds. This could be for weight initialization for SGD, e-greedy action selection, randomness in the environment dynamics etc.



## References

* Took a lot of help from https://github.com/lilianweng/deep-reinforcement-learning-gym. 
