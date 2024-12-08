import numpy as np
import torch
from tqdm import tqdm 
import statistics
from typing import Dict
from itertools import count

from models.agents import CartPoleAgent

def train(env, agent: CartPoleAgent, device: str, 
                            num_episodes: int,
                            patience: int = 50,
                            n_eps_mean_for_early_stopping: int = 100,
                            target_r = 490) -> Dict:
    """
    Train the agent with early stopping based on average reward over a window.
    
    Args:
        patience: Number of episodes to wait for improvement before stopping
        window_size: Size of window for calculating moving average
        target_reward: Target average reward to consider training successful
        verbose: Whether to show progress bar
    """
    tr_info = {'total_r': []}
    best_avg_reward = float('-inf')
    patience_counter = 0
    
    progressbar = tqdm(range(num_episodes)) 

    for i_episode in progressbar:
        total_reward = 0
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        done = False
        while True:
            if done:
                tr_info['total_r'].append(total_reward)
                break

            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            next_state = None if terminated else torch.tensor(
                observation, dtype=torch.float32, device=device).unsqueeze(0)

            agent.memorize_observation(state, action, next_state, reward)
            state = next_state
            agent.optimize()
            agent.update()

        if len(tr_info["total_r"]) >= n_eps_mean_for_early_stopping:
            mean = np.mean(tr_info['total_r'][-n_eps_mean_for_early_stopping:])

            if mean >= target_r:
                progressbar.set_postfix({
                    'episode_id': i_episode,
                    'total_r': total_reward,
                })

                progressbar.close()
                return tr_info
            

        progressbar.set_postfix({
            'episode_id': i_episode,
            'total_r': total_reward,
        })

    progressbar.close()

    return tr_info