"""
Neural network architectures for QMIX multi-agent system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingConvDQN(nn.Module):
    """Agent Network (estimates Q_i(o_i, u_i)) using Dueling DQN architecture."""

    def __init__(self, input_channels, grid_size, num_actions, feature_dim=2):
        super(DuelingConvDQN, self).__init__()
        self.grid_size = grid_size
        self.feature_dim = feature_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU()
        )

        # Calculate conv_out_size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, grid_size, grid_size)
            conv_out_size = self.conv_layers(dummy_input).view(1, -1).size(1)

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size + feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size + feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x, features):
        """
        Forward pass.
        Args:
            x: [batch, channels, H, W] - Grid input
            features: [batch, feature_dim] - Additional features (orientation)
        """
        batch_size = x.size(0)
        x = self.conv_layers(x)
        x = x.view(batch_size, -1)  # Flatten
        combined = torch.cat((x, features), dim=1)
        value = self.value_stream(combined)
        advantage = self.advantage_stream(combined)
        # Combine value and advantage streams
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class QMixNetwork(nn.Module):
    """QMIX Mixing Network that combines individual agent Q-values."""

    def __init__(self, num_agents, global_state_dim, mixing_embed_dim=64):
        super(QMixNetwork, self).__init__()
        self.num_agents = num_agents
        self.global_state_dim = global_state_dim
        self.embed_dim = mixing_embed_dim

        # Hypernetwork layers generating weights and biases based on global state
        self.hyper_w1 = nn.Sequential(
            nn.Linear(global_state_dim, self.embed_dim * self.num_agents)
        )
        self.hyper_b1 = nn.Linear(global_state_dim, self.embed_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(global_state_dim, self.embed_dim)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(global_state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        """
        Mix individual agent Q-values into Q_tot.
        Args:
            agent_qs: (batch_size, num_agents) - Q-values from individual agents
            states: (batch_size, global_state_dim) - Global state tensor
        Returns:
            q_tot: (batch_size, 1) - Mixed Q-value
        """
        batch_size = agent_qs.size(0)
        states = states.reshape(-1, self.global_state_dim)

        # Generate weights and biases from global state
        w1 = torch.abs(self.hyper_w1(states))  # Enforce positivity
        b1 = self.hyper_b1(states)
        w2 = torch.abs(self.hyper_w2(states))  # Enforce positivity
        b2 = self.hyper_b2(states)

        # Reshape for batch matrix multiplication
        w1 = w1.view(-1, self.num_agents, self.embed_dim)  # (batch, N, embed)
        b1 = b1.view(-1, 1, self.embed_dim)  # (batch, 1, embed)
        w2 = w2.view(-1, self.embed_dim, 1)  # (batch, embed, 1)
        b2 = b2.view(-1, 1, 1)  # (batch, 1, 1)

        # Reshape agent_qs for batch matrix multiplication: (batch, 1, N)
        q_vals_reshaped = agent_qs.view(-1, 1, self.num_agents)

        # First layer mixing
        hidden = F.elu(torch.bmm(q_vals_reshaped, w1) + b1)  # (batch, 1, embed)

        # Second layer mixing
        q_tot = torch.bmm(hidden, w2) + b2  # (batch, 1, 1)

        return q_tot.view(batch_size, 1)  # Return shape (batch_size, 1)
