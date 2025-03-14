# DQN and DDQN implementation on Gymnasium's CartPole environment
This is the implementation of DQN and DDQN on Gymnasium's CartPole environment.

## Prerequisite
This is the list of the versions that was used. Lower or higher version
does not guarantee bug-free execution of our code.
- python==3.10 
- CUDA version 12.4
- conda == 24.10

NOTE: We didn't use a Python version 3.10 specific version in our implementation
however, this is tied to other packages in conda. Therefore it is recommend to
use the same Python version.

## Install
We've provided a `environment.yml` file that can be installed with conda using
the following commands.
```
conda env create -f environment.yml
```

## Folder structure
The experiment and code run can be found in our `main.ipynb` file and the rest
of our implementation can be described in the following folder structure


```
.
├── models/
│   ├── agents/ 
│   │   ├── DDQL.py            # Double Deep Q-Learning agent
│   │   └── DQL.py             # Deep Q-Learning agent
│   └── networks/
│       └── DQN.py             # Deep Q-Network implementation
│
├── train/
│   ├── hyperparam_search_ddqn.py  # Hyperparameter optimization for DDQN
│   ├── hyperparam_search_dqn.py   # Hyperparameter optimization for DQN
│   ├── train_DDQL.py          # DDQN training script
│   ├── train_DQL.py           # DQN training script
│   ├── __init__.py            # Package initializer
│   ├── ReplayMemory.py        # Experience replay implementation
│   └── tuples.py              # Data structure definitions
│
├── best_ddqn.pkl              # Best DDQN model checkpoint
├── best_ddqn_conf.json        # Best DDQN configuration
├── best_dqn.pkl               # Best DQN model checkpoint
├── best_dqn_conf.json         # Best DQN configuration
├── environment.yml            # Conda environment specification
├── main.ipynb                 # Main experimentation notebook
├── README.md                  # Project documentation
└── run_sim.py                 # Simulation runner script
```

