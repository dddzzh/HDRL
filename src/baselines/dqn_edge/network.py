import torch
from torch import nn


class ObjectiveQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(state_dim), int(hidden_dim)), nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)), nn.ReLU(),
            nn.Linear(int(hidden_dim), int(action_dim)),
        )

    def forward(self, state):
        return self.model(state)


class ContextAttention(nn.Module):
    """Three FC layers followed by the paper's two-objective softmax."""

    def __init__(self, input_dim=4, hidden_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)), nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)), nn.ReLU(),
            nn.Linear(int(hidden_dim), 2),
        )

    def forward(self, context):
        return torch.softmax(self.model(context), dim=-1)
