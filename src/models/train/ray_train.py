import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
import gymnasium as gym
import torch

from models.agents.DQL import CartPoleDQNAgent

class RayTuneDQN:
    def __init__(self, num_samples=10, max_episodes=500):
        self.num_samples = num_samples
        self.max_episodes = max_episodes
        
    def train_with_config(self, config, checkpoint_dir=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        env = gym.make("CartPole-v1")
        
        # Initialize agent with tune config
        agent = CartPoleDQNAgent(
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
        
        total_reward = 0
        
        # Training loop with Ray Tune reporting
        for episode in range(self.max_episodes):
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
                
                agent.memorize_observation(state, action, next_state, reward)
                state = next_state
                agent.optimize()
                agent.update()

            train.report({"episode_reward":total_reward})
            
        # Report metrics to Ray Tune
        env.close()

    def get_config_space(self):
        config = {
            "memory_size": tune.choice([5000, 10000, 20000]),
            "batch_size": tune.choice([64, 128, 256]),
            "epsilon_start": tune.uniform(0.9, 1.0),
            "epsilon_min": tune.uniform(0.01, 0.1),
            "epsilon_decay": tune.choice([500, 1000, 2000]),
            "tau": tune.loguniform(1e-3, 1e-2),
            "gamma": tune.uniform(0.95, 0.99),
            "learning_rate": tune.loguniform(1e-4, 1e-3)
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
                metric="episode_reward",
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
        best_trial = analysis.get_best_trial("episode_reward", "max", "last")
        print("Best trial config:", best_trial.config)
        print("Best trial final reward:", best_trial.last_result["episode_reward"])
