"""
Baseline implementations for comparison with QMIX.

Baselines:
1. Greedy Frontier: Always move toward nearest uncovered cell
2. Independent Q-Learning: Each agent has its own Q-network (no mixing)
3. Random: Pure random exploration (sanity check)

Critical: All baselines use SAME observation space as QMIX (fair comparison).
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from typing import Dict, Tuple, List

from networks import DuelingConvDQN


class GreedyFrontierAgent:
    """Greedy baseline: Move toward nearest uncovered cell (frontier)."""
    
    def __init__(self, agent_id: int, environment):
        self.agent_id = agent_id
        self.env = environment
        
    def select_action(self, state_tensor) -> Tuple[int, int]:
        """Select action greedily toward nearest uncovered cell."""
        agent = self.env.agents[self.agent_id]
        current_pos = agent.state.position
        
        # Find uncovered cells in local map
        frontiers = []
        for node, data in agent.local_map.nodes(data=True):
            pc = data.get('pc', 0.0)
            node_type = data.get('type', 'free')
            
            # Uncovered free cells are frontiers
            if node_type == 'free' and pc < self.env.coverage_threshold:
                frontiers.append(node)
        
        if not frontiers:
            # No frontiers found - explore randomly
            valid_actions = self.env.get_valid_actions(self.agent_id)
            return random.choice(valid_actions) if valid_actions else (0, 0)
        
        # Find nearest frontier
        min_dist = float('inf')
        nearest_frontier = None
        for frontier in frontiers:
            dist = abs(frontier[0] - current_pos[0]) + abs(frontier[1] - current_pos[1])
            if dist < min_dist:
                min_dist = dist
                nearest_frontier = frontier
        
        # Move toward nearest frontier
        dr = nearest_frontier[0] - current_pos[0]
        dc = nearest_frontier[1] - current_pos[1]
        
        # Normalize to single step
        action_r = np.clip(dr, -1, 1)
        action_c = np.clip(dc, -1, 1)
        
        action = (action_r, action_c)
        
        # Validate action
        valid_actions = self.env.get_valid_actions(self.agent_id)
        if action in valid_actions:
            return action
        
        # If direct path blocked, try alternatives
        alternatives = [
            (action_r, 0),  # Move vertically only
            (0, action_c),  # Move horizontally only
            (action_r, -action_c if action_c != 0 else 0),  # Diagonal alternative
            (-action_r if action_r != 0 else 0, action_c),
        ]
        
        for alt in alternatives:
            if alt in valid_actions:
                return alt
        
        # Fallback: random valid action
        return random.choice(valid_actions) if valid_actions else (0, 0)


class IndependentQLearningAgent:
    """Independent Q-Learning: Each agent learns independently (no coordination).
    
    CRITICAL: Configs MUST match QMIX exactly for fair comparison.
    ONLY difference: no mixer network.
    """
    
    def __init__(
        self, 
        agent_id: int, 
        environment,
        lr: float = 5e-4,  # MATCH QMIX lr_agents
        gamma: float = 0.99,  # MATCH QMIX
        epsilon_start: float = 1.0,  # MATCH QMIX (not 0.1!)
        epsilon_end: float = 0.05,  # MATCH QMIX
        epsilon_decay: float = 0.99,  # MATCH QMIX (not 0.95!)
        memory_capacity: int = 50000,  # MATCH QMIX (not 10000!)
        batch_size: int = 64,  # MATCH QMIX (not 32!)
        use_soft_update: bool = True,  # MATCH QMIX
        soft_update_tau: float = 0.001,  # MATCH QMIX Tier 1
        target_update_freq: int = 200,  # MATCH QMIX (only used if soft_update=False)
        device: str = "cpu"
    ):
        self.agent_id = agent_id
        self.env = environment
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.use_soft_update = use_soft_update
        self.soft_update_tau = soft_update_tau
        self.target_update_freq = target_update_freq
        self.optimization_steps = 0
        self.device = torch.device(device)
        
        # Get network architecture from QMIX agent (fair comparison)
        marl_agent = environment.agents[agent_id]
        input_channels = marl_agent.input_channels  # Agent attribute, not network
        num_actions = len(marl_agent.actions)
        grid_size = environment.grid_size
        feature_dim = marl_agent.feature_dim
        
        # Independent Q-network (no mixer)
        self.policy_net = DuelingConvDQN(
            input_channels=input_channels,
            grid_size=grid_size,
            num_actions=num_actions,
            feature_dim=feature_dim
        ).to(self.device)
        
        self.target_net = DuelingConvDQN(
            input_channels=input_channels,
            grid_size=grid_size,
            num_actions=num_actions,
            feature_dim=feature_dim
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_capacity)
        
    def select_action(self, state_tensor) -> Tuple[int, int]:
        """Epsilon-greedy action selection."""
        agent = self.env.agents[self.agent_id]
        valid_actions = self.env.get_valid_actions(self.agent_id)
        
        if random.random() < self.epsilon:
            # Explore
            return random.choice(valid_actions) if valid_actions else (0, 0)
        else:
            # Exploit
            grid_tensor, feature_tensor = state_tensor
            
            with torch.no_grad():
                grid_tensor = grid_tensor.unsqueeze(0).to(self.device)
                feature_tensor = feature_tensor.unsqueeze(0).to(self.device)
                q_values = self.policy_net(grid_tensor, feature_tensor)
            
            # Mask invalid actions
            valid_action_indices = [agent.actions.index(a) for a in valid_actions]
            masked_q = q_values.clone()
            mask = torch.ones_like(masked_q) * float('-inf')
            mask[0, valid_action_indices] = 0
            masked_q = masked_q + mask
            
            action_idx = masked_q.argmax(1).item()
            return agent.actions[action_idx]
    
    def store_transition(self, state, action_idx, reward, next_state, done):
        """Store transition in replay buffer."""
        self.memory.append((state, action_idx, reward, next_state, done))
    
    def optimize(self):
        """Train independent Q-network (no coordination)."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, action_indices, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        grids = torch.stack([s[0] for s in states]).to(self.device)
        features = torch.stack([s[1] for s in states]).to(self.device)
        next_grids = torch.stack([s[0] for s in next_states]).to(self.device)
        next_features = torch.stack([s[1] for s in next_states]).to(self.device)
        
        actions = torch.tensor(action_indices, dtype=torch.long, device=self.device).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device).unsqueeze(-1)
        dones = torch.tensor(dones, dtype=torch.float, device=self.device).unsqueeze(-1)
        
        # Current Q-values
        current_q = self.policy_net(grids, features).gather(1, actions)
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_grids, next_features).max(1)[0].unsqueeze(-1)
            td_target = rewards + self.gamma * (1 - dones) * next_q
        
        # Huber loss (same as QMIX for fair comparison)
        loss = F.smooth_l1_loss(current_q, td_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        self.optimization_steps += 1
        
        # Update target network (MATCH QMIX behavior)
        if self.use_soft_update:
            # Soft update (Tier 1 fix): 0.1% per step
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(
                    self.soft_update_tau * policy_param.data + (1.0 - self.soft_update_tau) * target_param.data
                )
        else:
            # Hard update (original QMIX)
            if self.optimization_steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class RandomAgent:
    """Random baseline: Pure random exploration (sanity check)."""
    
    def __init__(self, agent_id: int, environment):
        self.agent_id = agent_id
        self.env = environment
    
    def select_action(self, state_tensor) -> Tuple[int, int]:
        """Random action selection from valid actions."""
        valid_actions = self.env.get_valid_actions(self.agent_id)
        return random.choice(valid_actions) if valid_actions else (0, 0)


class BaselineRunner:
    """Utility to run baseline experiments with same interface as QMIX."""
    
    def __init__(self, environment, baseline_type: str = "greedy"):
        """
        Args:
            environment: MARL_QMIX_Environment instance
            baseline_type: "greedy", "independent", or "random"
        """
        self.env = environment
        self.baseline_type = baseline_type
        self.agents = []
        
        if baseline_type == "greedy":
            self.agents = [GreedyFrontierAgent(i, environment) 
                          for i in range(environment.num_agents)]
        elif baseline_type == "independent":
            self.agents = [IndependentQLearningAgent(i, environment, device=environment.device.type)
                          for i in range(environment.num_agents)]
        elif baseline_type == "random":
            self.agents = [RandomAgent(i, environment)
                          for i in range(environment.num_agents)]
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    def select_actions(self) -> Dict[int, Tuple[int, int]]:
        """Get actions for all agents."""
        actions = {}
        for i, baseline_agent in enumerate(self.agents):
            marl_agent = self.env.agents[i]
            state_tensor = marl_agent.get_state_tensor()
            actions[i] = baseline_agent.select_action(state_tensor)
        return actions
    
    def train_step(self, transitions: Dict[int, Tuple]):
        """Train independent Q-learning agents (ignored for greedy/random)."""
        if self.baseline_type != "independent":
            return None
        
        losses = []
        for i, baseline_agent in enumerate(self.agents):
            if i in transitions:
                state, action, reward, next_state, done = transitions[i]
                
                # Get action index
                marl_agent = self.env.agents[i]
                try:
                    action_idx = marl_agent.actions.index(action)
                except ValueError:
                    action_idx = marl_agent.actions.index((0, 0))
                
                baseline_agent.store_transition(state, action_idx, reward, next_state, done)
                loss = baseline_agent.optimize()
                if loss is not None:
                    losses.append(loss)
        
        return np.mean(losses) if losses else None
    
    def decay_epsilon(self):
        """Decay epsilon for independent Q-learning agents."""
        if self.baseline_type == "independent":
            for agent in self.agents:
                agent.decay_epsilon()
    
    def get_epsilon(self) -> float:
        """Get average epsilon (for independent Q-learning)."""
        if self.baseline_type == "independent":
            return np.mean([agent.epsilon for agent in self.agents])
        return 0.0  # Greedy/random don't use epsilon
