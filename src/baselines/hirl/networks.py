from __future__ import annotations

import torch
from torch import nn


def mlp(input_dim, output_dim, hidden_dim, output_activation=None):
    layers = [
        nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    ]
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.q1 = mlp(state_dim + action_dim, 1, hidden_dim)
        self.q2 = mlp(state_dim + action_dim, 1, hidden_dim)

    def forward(self, state, action):
        value = torch.cat([state, action], dim=-1)
        return self.q1(value), self.q2(value)

    def first(self, state, action):
        return self.q1(torch.cat([state, action], dim=-1))
