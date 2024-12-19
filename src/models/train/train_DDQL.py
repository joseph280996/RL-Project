import torch
from tqdm import tqdm 
from typing import Dict

from models.agents.DDQL import CartPoleDDQNAgent

def train_DDQL(env, agent: CartPoleDDQNAgent, device: str, 
                            num_episodes: int) -> Dict:
    """
    Train the agent with early stopping based on average reward over a window.
    
    Args:
        env: The environment to use.
        agent: The agent to train (default: CartPoleDDQNAgent)
        device: The Pytorch device ('cpu', 'cuda', or 'mps')
        num_episodes: The number of episodes to run the training for.
    """
    tr_info = {'total_r': []}
    
    progressbar = tqdm(range(num_episodes))

    for i_episode in progressbar:
        total_reward = 0
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        done = False
        while True:
            if done:
                tr_info['total_r'].append(total_reward)
                progressbar.set_postfix({"total_r": total_reward})
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

    return tr_info

