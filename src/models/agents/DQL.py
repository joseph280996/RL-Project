import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

from models.tuples import Transition
from models.networks.DQN import DQN
from models.ReplayMemory import ReplayMemory

class CartPoleDQNAgent:
    def __init__(self, 
                 env, 
                 state_dim, 
                 action_dim, 
                 device,
                 memory_size = 10000,
                 batch_size = 128,
                 epsilon_start = 0.9,
                 epsilon_min = 0.05,
                 epsilon_decay = 1000,
                 tau = 0.005,
                 gamma = 0.99,
                 learning_rate = 1e-4):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env = env

        # Hyperparameters
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.memory = ReplayMemory(memory_size)

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True
        )
        self.criterion = nn.SmoothL1Loss()

        self.steps_done = 0

    def memorize_observation(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)


    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * math.exp(
            -1.0 * self.steps_done / self.epsilon_decay
        )

        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[self.env.action_space.sample()]], device=self.device, dtype=torch.long
            )

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Update the model's weight
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step() 

    def update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
