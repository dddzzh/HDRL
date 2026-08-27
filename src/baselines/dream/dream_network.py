import torch
from torch import nn


class DReaMQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(int(state_dim), int(hidden_dim))
        self.bn1 = nn.BatchNorm1d(int(hidden_dim))
        self.fc2 = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.bn2 = nn.BatchNorm1d(int(hidden_dim))
        self.out = nn.Linear(int(hidden_dim), int(action_dim))

    def forward(self, state):
        # Normalize the affine activations before ReLU.  ReLU-before-BN made
        # permanently inactive units accumulate an almost-zero running
        # variance; a previously unseen activation was then amplified by
        # orders of magnitude during deterministic evaluation.
        x = torch.relu(self.bn1(self.fc1(state)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return self.out(x)
