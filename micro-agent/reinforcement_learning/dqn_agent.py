import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from reinforcement_learning.q_network import QNetwork
from reinforcement_learning.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 10000,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: str = None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon-greedy exploration parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Auto-detect GPU if available
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize Q-network and target network with same weights
        self.q_network = QNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity, state_dim)
        # Preallocate buffer to avoid repeated allocation during inference
        self._state_buffer = torch.zeros((1, state_dim), dtype=torch.float32, device=self.device)
        self.loss_fn = nn.MSELoss()
    
    def get_epsilon(self) -> float:
        # Exponentially decay exploration rate over training steps
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-self.steps_done / self.epsilon_decay)

    def select_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> int:
        # Epsilon-greedy: explore randomly or exploit Q-values
        if training and random.random() < self.get_epsilon():
            return random.randrange(self.n_actions)

        # Greedy action selection using Q-network
        with torch.no_grad():
            self._state_buffer[0] = torch.from_numpy(state)
            q_values = self.q_network(self._state_buffer)
            return int(q_values.argmax(dim=1)[0])
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        self.buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> float:
        # Wait for sufficient experience before training
        if len(self.buffer) < self.batch_size:
            return 0.0

        # Sample random minibatch from replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Transfer batch to GPU asynchronously for performance
        states = torch.from_numpy(states).float().to(self.device, non_blocking=True)
        actions = torch.from_numpy(actions).long().to(self.device, non_blocking=True)
        rewards = torch.from_numpy(rewards).float().to(self.device, non_blocking=True)
        next_states = torch.from_numpy(next_states).float().to(self.device, non_blocking=True)
        dones = torch.from_numpy(dones).float().to(self.device, non_blocking=True)

        self.optimizer.zero_grad()

        # Get Q-values for taken actions
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute TD target using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values)

        loss.backward()
        self.optimizer.step()

        self.steps_done += 1

        # Periodically sync target network to stabilize learning
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()
    
    def save(
        self,
        path: str
    ):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
        }, path)
    
    def load(
        self,
        path: str
    ):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']

    def save_state(self) -> dict:
        # Save DQN state to dict (in-memory, no disk I/O)
        return {
            'q_network': {k: v.cpu().clone() for k, v in self.q_network.state_dict().items()},
            'target_network': {k: v.cpu().clone() for k, v in self.target_network.state_dict().items()},
            'optimizer': {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items()},
            'steps_done': self.steps_done,
        }

    def restore_state(self, state: dict) -> None:
        # Restore DQN state from dict
        q_net_state = {k: v.to(self.device) for k, v in state['q_network'].items()}
        target_net_state = {k: v.to(self.device) for k, v in state['target_network'].items()}
        opt_state = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in state['optimizer'].items()}

        self.q_network.load_state_dict(q_net_state)
        self.target_network.load_state_dict(target_net_state)
        self.optimizer.load_state_dict(opt_state)
        self.steps_done = state['steps_done']