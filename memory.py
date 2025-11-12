"""
Replay memory implementation for QMIX.
"""

import random
import torch
from collections import deque


class QMixReplayMemory:
    """Replay memory for storing and sampling joint transitions in QMIX."""

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, global_state, agent_states, agent_actions_indices, reward,
             next_global_state, next_agent_states, done):
        """Save a joint transition for QMIX. Detach tensors."""
        # Store tensors on CPU to save GPU memory
        global_state_cpu = global_state.detach().cpu() if isinstance(global_state, torch.Tensor) else global_state
        agent_states_cpu = {aid: (s[0].detach().cpu(), s[1].detach().cpu())
                           for aid, s in agent_states.items()}
        next_global_state_cpu = next_global_state.detach().cpu() if isinstance(next_global_state, torch.Tensor) else next_global_state
        next_agent_states_cpu = {aid: (s[0].detach().cpu(), s[1].detach().cpu())
                                for aid, s in next_agent_states.items()}

        self.memory.append((global_state_cpu, agent_states_cpu, agent_actions_indices, reward,
                            next_global_state_cpu, next_agent_states_cpu, done))

    def sample(self, batch_size):
        """Sample a batch of joint transitions."""
        actual_batch_size = min(batch_size, len(self.memory))
        return random.sample(self.memory, actual_batch_size)

    def __len__(self):
        return len(self.memory)
