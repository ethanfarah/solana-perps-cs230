import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        # Two-layer feedforward network for Q-value estimation
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.net(x)