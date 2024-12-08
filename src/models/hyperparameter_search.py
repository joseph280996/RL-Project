import pandas as pd
import torch
import gymnasium as gym
from itertools import product
from typing import Dict, Any

from models.agents import *
from models.train import *
from IPython.display import display

class GridHyperparameterSearch:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else "cpu")
        self.results = []
        
    def evaluate_config(self, config: Dict[str, Any]):
        """Evaluate a single hyperparameter configuration"""
        env = gym.make('CartPole-v1')
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        agent = CartPoleAgent(
            env=env,
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
            batch_size=config["batch_size"],
            epsilon_start=config["epsilon_start"],
            epsilon_min=config["epsilon_min"],
            epsilon_decay=config["epsilon_decay"],
            tau=config["tau"],
            gamma=config["gamma"],
            learning_rate=config["learning_rate"],
            memory_size=config["memory_size"]
        )
        
        tr_info = train(
            env=env,
            agent=agent,
            device=self.device,
            num_episodes=config["n_epochs"],
        )
        
        # Calculate metrics for speed and performance
        eps_to_converge = len(tr_info['total_r'])
        
        result = {
            **config, 
            "eps_to_converge": eps_to_converge
        }
        self.results.append(result)
        return result
        
    def run_search(self):
        """Run grid search for hyperparameters"""
        # Define parameter grid
        param_grid = {
            "n_epochs": [1000],
            "batch_size": [128],
            "epsilon_start": [0.9, 1],
            "epsilon_min": [0.05, 0.005],
            "epsilon_decay": [1000],
            "tau": [0.001, 0.005, 0.01],
            "gamma": [0.9, 0.95, 0.99],
            "learning_rate": [1e-4, 1e-3, 1e-2],
            "memory_size": [10000, 100000, 1000000]
        }
        
        # Generate all combinations
        keys, values = zip(*param_grid.items())
        configurations = [dict(zip(keys, v)) for v in product(*values)]
        
        # Run grid search with progress bar
        print(f"Testing {len(configurations)} configurations...")
        for config in tqdm(configurations):
            self.evaluate_config(config)
        
        # Convert results to DataFrame and sort by score (reward/episodes)
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('eps_to_converge', ascending=False)
        
        # Display top results
        print("\nTop configurations (higher score means better reward achieved faster):")
        display(results_df[['batch_size', 'epsilon_start', 'epsilon_min', 'epsilon_decay', 
                          'tau', 'gamma', 'learning_rate','eps_to_converge']].head())
        
        best_config = results_df.iloc[0].drop(['eps_to_converge']).to_dict()
        print("\nBest hyperparameters found:", best_config)
        return best_config, results_df
