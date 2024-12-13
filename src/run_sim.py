import torch
import argparse
import pickle
import json
import gymnasium as gym

from models import CartPoleDQNAgent, CartPoleDDQNAgent

parser = argparse.ArgumentParser(description='Load DQN or DDQN model from pickle file')
parser.add_argument('--model', type=str, required=True, choices=['dqn', 'ddqn'],
                    help='Type of model to load (dqn or ddqn)')
parser.add_argument('--initial_state', action='store_true',
                    help='Flag to indicate initial state (default: False)')
    

def load_model(model_type):
    """
    Load the appropriate pickle file based on model type.
    
    Args:
        model_type (str): Type of model ('dqn' or 'ddqn')
        
    Returns:
        object: Loaded model from pickle file
    """
    if model_type.lower() not in ['dqn', 'ddqn']:
        raise ValueError("Model type must be either 'dqn' or 'ddqn'")
    
    filename = f"best_{model_type.lower()}.pkl"
    
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Successfully loaded model from {filename}")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {filename}. Please ensure the file exists in the current directory.")

def create_initial_state_mode(model_type, env, device):
    model_type = model_type.lower()
    with open(f'best_{model_type}_conf.json', 'r') as f:
        config = json.load(f)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if model_type.lower() not in ['dqn', 'ddqn']:
        raise ValueError("Model type must be either 'dqn' or 'ddqn'")

    if model_type == 'dqn':
        return CartPoleDQNAgent(env, state_dim, action_dim, device, 
                                    **config)

    return CartPoleDDQNAgent(env, state_dim, action_dim, device, 
                                    **config)

if __name__ == "__main__":
    args = parser.parse_args()

    env = gym.make("CartPole-v1", render_mode='human')
    env.reset()

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

    use_initial_state = args.initial_state
    if not use_initial_state:
        agent = load_model(args.model)    
    else:
        agent = create_initial_state_mode(args.model, env, device)
    
    total_reward = 0
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    done = False
    while not done:
        action = agent.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        next_state = None if terminated else torch.tensor(
            observation, dtype=torch.float32, device=device).unsqueeze(0)

        state = next_state
    
    print(total_reward)


