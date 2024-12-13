import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
import gymnasium as gym
import torch

from models.agents.DDQL import CartPoleDDQNAgent

class RayTuneDDQN:
    def __init__(self, num_samples=10, max_episodes=500):
        self.num_samples = num_samples
        self.max_episodes = max_episodes

    def train_with_config(self, config, checkpoint_dir=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        env = gym.make("CartPole-v1")

        # Initialize agent with tune config
        agent = CartPoleDDQNAgent(
            env=env,
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            device=device,
            memory_size=config["memory_size"],
            batch_size=config["batch_size"],
            epsilon_start=config["epsilon_start"],
            epsilon_min=config["epsilon_min"],
            epsilon_decay=config["epsilon_decay"],
            tau=config["tau"],
            gamma=config["gamma"],
            learning_rate=config["learning_rate"]
        )

        episode_rewards = []
        window_size = 100

        # Training loop with Ray Tune reporting
        for episode in range(self.max_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            ep_reward = 0

            done = False
            while not done:
                action = agent.select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                ep_reward += reward
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                next_state = None if terminated else torch.tensor(
                    observation, dtype=torch.float32, device=device).unsqueeze(0)

                agent.memorize_observation(state, action, next_state, reward)
                state = next_state
                agent.optimize()
                agent.update()

            episode_rewards.append(ep_reward)

            if len(episode_rewards) >= window_size:
                moving_avg = sum(episode_rewards[-window_size:]) / window_size
                train.report({
                    "episode_reward":ep_reward,
                    "moving_average_reward": moving_avg,
                    "stability_score": min(episode_rewards[-window_size:])
                })

        # Report metrics to Ray Tune
        env.close()

    def get_config_space(self):
        config = {
            "memory_size": tune.choice([10000, 20000, 50000]),  # Increased upper range
            "batch_size": tune.choice([32, 64, 128]),  # Added smaller batch size
            "epsilon_start": tune.uniform(0.95, 1.0),  # Narrowed to ensure good initial exploration
            "epsilon_min": tune.loguniform(1e-3, 5e-2),  # Log scale for better resolution
            "epsilon_decay": tune.qlograndint(1000, 10000, q=500),  # Log scale with quantization
            "tau": tune.loguniform(1e-3, 5e-2),  # Wider range for target network updates
            "gamma": tune.uniform(0.97, 0.999),  # Narrowed to focus on long-term rewards
            "learning_rate": tune.loguniform(1e-4, 5e-3),  # Slightly wider range
        }
        return config

    def run_hyperparameter_search(self):
        ray.init()

        scheduler = ASHAScheduler(
            max_t=self.max_episodes,
            grace_period=100,
            reduction_factor=2
        )

        tuner = tune.Tuner(
            tune.with_resources(
                self.train_with_config,
                {"gpu": 0.1, "cpu": 1}),
            tune_config=tune.TuneConfig(
                metric="moving_average_reward",
                mode="max",
                scheduler=scheduler,
                num_samples=self.num_samples,
                max_concurrent_trials=20
            ),
            param_space=self.get_config_space(),
        )
        result = tuner.fit()

        ray.shutdown()
        return result

    def print_best_config(self, analysis):
        # Convert the ResultGrid to a DataFrame
        df = analysis.get_dataframe()
        
        # First, filter for trials with top 25% moving average reward
        reward_threshold = df['moving_average_reward'].quantile(0.75)
        top_performers = df[df['moving_average_reward'] >= reward_threshold]
        
        # Among these top performers, find the one with the best stability score
        best_trial = top_performers.loc[top_performers['stability_score'].idxmax()]
        
        print("\nBest Trial Performance:")
        print(f"Moving Average Reward (last 100 episodes): {best_trial['moving_average_reward']:.2f}")
        print(f"Stability Score: {best_trial['stability_score']:.2f}")
        
        print("\nBest Configuration:")
        params = ['memory_size', 'batch_size', 'epsilon_start', 'epsilon_min', 
                'epsilon_decay', 'tau', 'gamma', 'learning_rate']
        best_trial = best_trial.to_dict()
        best_config = {}
        for param in params:
            best_config[param] = best_trial[f'config/{param}']
            print(f"{param}: {best_trial[f'config/{param}']}")

        return best_config