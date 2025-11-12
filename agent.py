"""
Multi-Agent RL Coverage Agent implementation.
"""

import math
import random
import copy
import numpy as np
import networkx as nx
import torch

from data_structures import RobotState, CommunicationEvent
from networks import DuelingConvDQN


class MARLCoverageAgent:
    """Multi-Agent Reinforcement Learning Coverage Agent."""

    def __init__(
        self,
        agent_id: int,
        initial_state: RobotState,
        environment,
        grid_size: int,
        sensor_range: int,
        comm_range: float = 5.0,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        use_dueling: bool = True,
        device: str = "cpu",
    ):
        self.agent_id = agent_id
        self.state = initial_state
        self.env = environment
        self.grid_size = grid_size
        self.sensor_range = sensor_range
        self.comm_range = comm_range
        self.device = torch.device(device)
        self.steps_done = 0

        # Epsilon parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start

        # Network type flag
        self.use_dueling = use_dueling

        # Action space definition
        self.actions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 0), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        self.action_size = len(self.actions)

        # Network input dimensions
        self.input_channels = 2
        self.feature_dim = 2

        # Initialize agent's policy and target networks
        NetworkClass = DuelingConvDQN
        self.policy_net = NetworkClass(
            self.input_channels, grid_size, self.action_size, self.feature_dim
        ).to(self.device)
        self.target_net = NetworkClass(
            self.input_channels, grid_size, self.action_size, self.feature_dim
        ).to(self.device)

        # Sync target net initially (will be updated centrally)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only for inference

        # Local knowledge attributes
        self.local_map = nx.Graph()
        self.known_positions = {}
        self.communication_history = []
        self.priority = random.random()  # Static priority for conflict resolution

    def _initialize_local_map(self):
        """Initialize agent's local map based on the current global map."""
        self.local_map = nx.Graph()
        if self.env and self.env.world_state and self.env.world_state.graph:
            self.local_map = self.env.world_state.graph.copy()
        else:
            print(f"Warning: Agent {self.agent_id} could not initialize local map.")

    def update_local_map(self, node, pc_value, node_type=None):
        """Update local map with new coverage or obstacle information.

        CRITICAL POMDP FIX: Properly record discovered obstacles.
        """
        if not self.local_map.has_node(node):
            return False

        updated = False
        current_type = self.local_map.nodes[node].get('type', 'free')

        # ✅ POMDP FIX: Handle obstacle discovery
        if node_type == 'occupied':
            if current_type != 'occupied':
                self.local_map.nodes[node]['type'] = 'occupied'
                self.local_map.nodes[node]['pc'] = 0.0  # Obstacles are not coverable
                updated = True
            return updated  # Don't update coverage for obstacles

        # Update coverage probability (only for non-obstacles)
        current_pc = self.local_map.nodes[node].get('pc', -1.0)
        if pc_value > current_pc:
            self.local_map.nodes[node]['pc'] = pc_value
            updated = True
            if pc_value >= self.env.coverage_threshold:
                self.local_map.nodes[node]['type'] = 'covered'

        # Update type if specified
        if node_type == 'covered' and current_type != 'covered' and current_type != 'occupied':
            self.local_map.nodes[node]['type'] = 'covered'
            updated = True

        return updated

    def communicate(self, other_agent, env_time):
        """Exchange local map information with another agent if within communication range."""
        if self._is_within_comm_range(other_agent):
            map_updated = self._exchange_map_info(other_agent)
            self.known_positions[other_agent.agent_id] = other_agent.state.position
            other_agent.known_positions[self.agent_id] = self.state.position
            if map_updated:
                event = CommunicationEvent(
                    sender_id=self.agent_id,
                    receiver_id=other_agent.agent_id,
                    data_type='map_exchange',
                    timestamp=env_time
                )
                self.communication_history.append(event)
                other_agent.communication_history.append(copy.deepcopy(event))
                return True
        return False

    def _is_within_comm_range(self, other_agent):
        """Check if another agent is within communication range."""
        dist_sq = sum((p1 - p2)**2 for p1, p2 in zip(self.state.position, other_agent.state.position))
        return dist_sq <= self.comm_range**2

    def _exchange_map_info(self, other_agent):
        """Merge knowledge from other agent's local map into this agent's map."""
        updates_made_by_me = False
        updates_made_by_other = False
        nodes_to_process = list(other_agent.local_map.nodes())
        for node in nodes_to_process:
            other_data = other_agent.local_map.nodes[node]
            other_pc = other_data.get('pc', 0.0)
            other_type = other_data.get('type', None)
            updated_me = self.update_local_map(node, other_pc, other_type)
            updates_made_by_me |= updated_me
            if self.local_map.has_node(node):
                my_data = self.local_map.nodes[node]
                my_pc = my_data.get('pc', 0.0)
                my_type = my_data.get('type', None)
                updated_other = other_agent.update_local_map(node, my_pc, my_type)
                updates_made_by_other |= updated_other
        return updates_made_by_me or updates_made_by_other

    def get_state_tensor(self):
        """Create state tensors (coverage grid, obstacle grid, features) for the NN input.

        CRITICAL: Agent sees ONLY what it has observed through sensors (TRUE POMDP).
        Obstacle grid is built from local_map, NOT from environment's ground truth.
        """
        coverage_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        obstacle_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)  # ✅ Build from local knowledge

        # Build from local_map ONLY (no omniscient knowledge)
        for node, data in self.local_map.nodes(data=True):
            r, c = node
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                # Coverage probability
                coverage_grid[r, c] = data.get('pc', 0.0)

                # ✅ POMDP FIX: Only known obstacles (discovered via raycasting)
                if data.get('type') == 'occupied':
                    obstacle_grid[r, c] = 1.0

        grid_stack = np.stack([coverage_grid, obstacle_grid], axis=0)
        grid_tensor = torch.from_numpy(grid_stack).to(self.device)  # Shape [C, H, W]
        orientation_features = np.array([math.sin(self.state.orientation), math.cos(self.state.orientation)], dtype=np.float32)
        feature_tensor = torch.from_numpy(orientation_features).to(self.device)  # Shape [FeatDim]
        return grid_tensor, feature_tensor

    def select_action(self, state, valid_actions):
        """Select an action using epsilon-greedy strategy based on the current state and valid actions."""
        self.steps_done += 1
        if not valid_actions:
            return (0, 0)  # Default to staying still if trapped

        sample = random.random()
        if sample < self.epsilon:
            action = random.choice(valid_actions)
        else:
            grid_tensor, feature_tensor = state
            with torch.no_grad():
                # Add batch dimension for network input
                grid_batch = grid_tensor.unsqueeze(0)
                feature_batch = feature_tensor.unsqueeze(0)
                q_values = self.policy_net(grid_batch, feature_batch)
                q_values_np = q_values.squeeze(0).cpu().numpy()

            # Log individual Q-values for analysis
            self.env.metrics.agent_q_values[self.agent_id].append(q_values_np.tolist())

            best_q = -float('inf')
            best_action = valid_actions[0]  # Default to first valid action
            found_valid_q = False

            # Find the valid action with the highest Q-value
            for act in valid_actions:
                try:
                    idx = self.actions.index(act)
                    if 0 <= idx < len(q_values_np):
                        if q_values_np[idx] > best_q:
                            best_q = q_values_np[idx]
                            best_action = act
                            found_valid_q = True
                    else:
                        print(f"Error Agent {self.agent_id}: Action index {idx} out of bounds for Q-values.")
                except ValueError:
                    print(f"Error Agent {self.agent_id}: Action {act} not in standard action list.")

            if not found_valid_q:
                action = random.choice(valid_actions)
            else:
                action = best_action
        return action

    def update_epsilon(self):
        """Decay exploration rate epsilon based on total steps done."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_state(self, new_state):
        """Update the agent's internal knowledge of its own state."""
        self.state = new_state

    def save_agent_state(self):
        """Returns agent-specific state (epsilon, steps, networks)."""
        return {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }

    def load_agent_state(self, checkpoint):
        """Loads agent-specific state (epsilon, steps, networks)."""
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.steps_done = checkpoint.get('steps_done', 0)
        self.target_net.eval()  # Ensure target net is in eval mode
