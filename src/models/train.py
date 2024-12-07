import numpy as np
from itertools import count
from models.agents import CartPoleAgent
import torch
from tqdm import tqdm 

def train(env, agent:CartPoleAgent, device, num_episodes):
    tr_info = {'durations': [], 'total_rewards': []}

    progress_bar = tqdm(range(num_episodes))
    pbar_info = {}


    for i_episode in progress_bar:
        # Initialize the environment and get its state
        total_reward = 0
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            agent.memorize_observation(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.optimize()

            # Perform policy update
            agent.update()
            
            if done:
                tr_info['durations'].append(t + 1)
                tr_info['total_rewards'].append(total_reward)
                pbar_info.update({
                    'total_r': total_reward,
                    'duration': t + 1
                })
                break

        progress_bar.set_postfix(pbar_info)

    progress_bar.close()
    return tr_info
