"""
Multi-Agent RL QMIX Environment for coverage tasks.
"""

import os
import time
import math
import random
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Set, Optional

from data_structures import RobotState, WorldState, CoverageMetrics
from agent import MARLCoverageAgent
from networks import QMixNetwork
from memory import QMixReplayMemory
from utils import (
    ensure_dir, generate_room_map, generate_cave_map,
    generate_empty_map, generate_random_map
)
from visualization import (
    visualize_coverage_plot, visualize_agent_local_map_plot,
    visualize_agent_local_timesteps_animation, visualize_evaluation_timesteps_animation,
    visualize_learning_metrics_dashboard
)


class MARL_QMIX_Environment:
    """Multi-Agent Reinforcement Learning Environment using QMIX."""

    def __init__(
        self,
        grid_size: int = 20,
        num_agents: int = 4,
        sensor_range: int = 5,
        comm_range: float = 7.0,
        coverage_threshold: float = 0.8,
        completion_threshold_perc: float = 95.0,
        max_episodes: int = 300,
        max_steps_per_episode: int = 100,
        gamma_coverage: float = 15.0,
        step_penalty: float = -0.02,
        orientation_cost_factor: float = 0.02,
        invalid_move_penalty: float = -0.5,
        fov_degrees: float = 120.0,
        device: str = "cpu",
        use_dueling: bool = True,
        tensorboard_dir: str = "./runs/qmix_coverage",
        gamma: float = 0.99,
        batch_size: int = 64,
        memory_capacity: int = 50000,
        mixer_embed_dim: int = 64,
        lr: float = 5e-4,
        target_update_freq: int = 200,
        use_soft_update: bool = True,
        soft_update_tau: float = 0.005,
        agent_config: dict = {},
        coverage_r0_factor: float = 2.5,
        coverage_k: float = 2.0
    ):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.sensor_range = sensor_range
        self.comm_range = comm_range
        self.coverage_threshold = coverage_threshold
        self.completion_threshold_perc = completion_threshold_perc
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode

        # Reward Params
        self.gamma_coverage = gamma_coverage
        self.step_penalty = step_penalty
        self.orientation_cost_factor = orientation_cost_factor
        self.invalid_move_penalty = invalid_move_penalty
        self.fov_radians = math.radians(fov_degrees)
        self.device = torch.device(device)
        self.use_dueling = use_dueling

        # Training Params
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_soft_update = use_soft_update
        self.soft_update_tau = soft_update_tau
        self.global_optimization_steps = 0
        self.agent_config = agent_config
        if 'use_dueling' in self.agent_config:
            del self.agent_config['use_dueling']

        # Coverage params
        self.coverage_r0 = self.sensor_range / coverage_r0_factor
        self.coverage_k = coverage_k

        # Tensorboard
        self.writer = None
        if tensorboard_dir:
            ensure_dir(tensorboard_dir)
            self.writer = SummaryWriter(tensorboard_dir)

        # Map Generation
        self.training_map_generators = {
            'empty': lambda: generate_empty_map(self.grid_size),
            'cave': lambda: generate_cave_map(self.grid_size),
            'room': lambda: generate_room_map(self.grid_size),
            'random': lambda: generate_random_map(self.grid_size)
        }
        self.training_map_order = ['room', 'cave', 'random']
        self.current_training_map_index = 0
        self.mode = 'train'

        # Environment State
        self.obstacle_grid = None
        self.world_state = None
        self.agent_positions = {}
        self.agents = []

        # Metrics & Time
        self.metrics = CoverageMetrics()
        self.env_time = 0.0
        self.previous_global_coverage = 0.0
        self.current_global_coverage = 0.0
        self.global_coverage_increase = 0.0
        self.current_rewards = {}

        # Centralized Components
        self.mixer = None
        self.target_mixer = None
        self.optimizer = None
        self.memory = QMixReplayMemory(capacity=memory_capacity)
        self.mixer_embed_dim = mixer_embed_dim
        self.lr = lr

        self.raycasting_cache = {}
        print(f"QMIX Environment initialized on device: {self.device}")

    def _initialize_world_state(self):
        """Initialize graph (only traversable nodes) and robot states based on self.obstacle_grid."""
        if self.obstacle_grid is None:
            return None
        G_base = nx.grid_2d_graph(self.grid_size, self.grid_size)
        G = nx.Graph()
        nodes_to_remove = []
        for node in G_base.nodes():
            r, c = node
            if self.obstacle_grid[r, c] == 1.0:
                nodes_to_remove.append(node)
            else:
                G.add_node(node, type='free', pc=0.0)
        for u, v in G_base.edges():
            if u not in nodes_to_remove and v not in nodes_to_remove:
                G.add_edge(u, v)
        for r in range(self.grid_size - 1):
            for c in range(self.grid_size - 1):
                n1, n2, n3, n4 = (r, c), (r + 1, c + 1), (r + 1, c), (r, c + 1)
                if G.has_node(n1) and G.has_node(n2) and G.has_node(n3) and G.has_node(n4):
                    G.add_edge(n1, n2)
                n1, n2 = (r + 1, c), (r, c + 1)
                if G.has_node(n1) and G.has_node(n2) and G.has_node(n3) and G.has_node(n4):
                    G.add_edge(n1, n2)
        robots = []
        placed_positions = set()
        free_positions = list(G.nodes())
        if not free_positions:
            return None
        for i in range(self.num_agents):
            pos = None
            corner_prefs = [
                (1, 1), (1, self.grid_size - 2),
                (self.grid_size - 2, 1), (self.grid_size - 2, self.grid_size - 2)
            ]
            random.shuffle(corner_prefs)
            for pref_pos in corner_prefs:
                if G.has_node(pref_pos) and pref_pos not in placed_positions:
                    pos = pref_pos
                    break
            if pos is None:
                available_free = [p for p in free_positions if p not in placed_positions]
                pos = random.choice(available_free) if available_free else random.choice(free_positions)
            placed_positions.add(pos)
            initial_orientation = random.uniform(0, 2 * math.pi)
            robots.append(RobotState(position=pos, orientation=initial_orientation))
        return WorldState(graph=G, robots=robots)

    def _create_agents(self):
        """Create agent instances using current world state robots and config."""
        if not self.world_state or not self.world_state.robots:
            return []
        agents = []
        for i, robot_state in enumerate(self.world_state.robots):
            if i >= self.num_agents:
                break
            agent = MARLCoverageAgent(
                agent_id=i,
                initial_state=robot_state,
                environment=self,
                grid_size=self.grid_size,
                sensor_range=self.sensor_range,
                comm_range=self.comm_range,
                use_dueling=self.use_dueling,
                device=self.device.type,
                **self.agent_config
            )
            agents.append(agent)
        return agents

    def get_global_state_tensor(self):
        """Creates a global state tensor for the QMIX mixer network.

        CRITICAL POMDP INTEGRITY: Uses ONLY shared knowledge from agent local maps.
        Does NOT use self.obstacle_grid (ground truth). Global state is built by
        aggregating what agents have collectively discovered through sensors.
        """
        global_coverage_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        global_obstacle_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # ✅ Aggregate from agent local maps ONLY (no ground truth leakage)
        for agent in self.agents:
            for node, data in agent.local_map.nodes(data=True):
                r, c = node
                if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                    # Max coverage across agents
                    global_coverage_grid[r, c] = max(global_coverage_grid[r, c], data.get('pc', 0.0))

                    # Union of discovered obstacles
                    if data.get('type') == 'occupied':
                        global_obstacle_grid[r, c] = 1.0

        agent_pos_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for agent_id, pos in self.agent_positions.items():
            if 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size:
                agent_pos_grid[pos[0], pos[1]] = 1.0

        flat_coverage = global_coverage_grid.flatten()
        flat_obstacles = global_obstacle_grid.flatten()
        flat_agent_pos = agent_pos_grid.flatten()

        orientations = []
        for i in range(self.num_agents):
            orientation = self.agents[i].state.orientation
            orientations.extend([math.sin(orientation), math.cos(orientation)])
        flat_orientations = np.array(orientations, dtype=np.float32)

        global_state_np = np.concatenate([flat_coverage, flat_obstacles, flat_agent_pos, flat_orientations])
        return torch.FloatTensor(global_state_np).to(self.device)

    def reset(self, full_reset=False, mode='train'):
        """Reset environment: generate map, initialize world state and agents."""
        self.mode = mode
        current_time_seed = int(time.time() * 1000) % (2**32)
        np.random.seed(current_time_seed)
        random.seed(current_time_seed)
        map_type = 'random'
        if self.mode == 'train':
            map_type = self.training_map_order[self.current_training_map_index]
        map_generator = self.training_map_generators[map_type]
        self.obstacle_grid = map_generator()
        self.world_state = self._initialize_world_state()
        if not self.world_state:
            raise RuntimeError("Failed to initialize world state.")
        self.agent_positions = {i: r.position for i, r in enumerate(self.world_state.robots)}

        if not self.agents or full_reset:
            self.agents = self._create_agents()
            if not self.agents:
                raise RuntimeError("Failed to create agents.")
            if self.mixer is None:
                _dummy_state = self.get_global_state_tensor()
                self.global_state_size = _dummy_state.shape[0]
                self.mixer = QMixNetwork(self.num_agents, self.global_state_size, self.mixer_embed_dim).to(self.device)
                self.target_mixer = QMixNetwork(self.num_agents, self.global_state_size, self.mixer_embed_dim).to(self.device)
                self.target_mixer.load_state_dict(self.mixer.state_dict())
                self.target_mixer.eval()
                all_params = list(self.mixer.parameters())
                for agent in self.agents:
                    all_params.extend(list(agent.policy_net.parameters()))
                self.optimizer = optim.Adam(all_params, lr=self.lr)

        for i, agent in enumerate(self.agents):
            if i < len(self.world_state.robots):
                agent.update_state(self.world_state.robots[i])
                agent._initialize_local_map()
                agent.known_positions = {}
                agent.communication_history = []
                if self.mode == 'train' or full_reset:
                    agent.epsilon = agent.epsilon_start
            else:
                print(f"Warning: Agent index {i} out of bounds.")

        self.metrics = CoverageMetrics()
        self.env_time = 0.0
        self.previous_global_coverage = 0.0
        for r_state in self.world_state.robots:
            if self.world_state.graph.has_node(r_state.position):
                self.world_state.graph.nodes[r_state.position]['pc'] = 1.0
                if 1.0 >= self.coverage_threshold:
                    self.world_state.graph.nodes[r_state.position]['type'] = 'covered'
                self.previous_global_coverage += 1.0
        self.current_global_coverage = self.previous_global_coverage
        self.global_coverage_increase = 0.0
        self.current_rewards = {i: 0.0 for i in range(self.num_agents)}
        self.raycasting_cache = {}
        for agent_id, agent in enumerate(self.agents):
            self.metrics.epsilon_values[agent_id].append(agent.epsilon)

        initial_states = {i: agent.get_state_tensor() for i, agent in enumerate(self.agents)}
        return initial_states

    def get_valid_actions(self, agent_id: int):
        """Get valid discrete actions (moves) for the agent."""
        valid_actions = []
        current_pos = self.agent_positions[agent_id]
        possible_actions = self.agents[agent_id].actions
        for action in possible_actions:
            target_pos = (current_pos[0] + action[0], current_pos[1] + action[1])
            if self.is_valid_position(target_pos, agent_id, current_pos, check_other_agents=False):
                valid_actions.append(action)
        if not valid_actions:
            return [(0, 0)]
        if (0, 0) not in valid_actions and self.is_valid_position(current_pos, agent_id, current_pos, check_other_agents=False):
            valid_actions.append((0, 0))
        return valid_actions

    def is_valid_position(self, pos, agent_id, current_pos, check_other_agents=True):
        """Check if a target position 'pos' is valid for movement."""
        if not self.is_in_bounds(pos):
            return False
        if self.obstacle_grid[pos[0], pos[1]] == 1.0:
            return False
        if not self.world_state or not self.world_state.graph.has_node(pos):
            return False
        if check_other_agents:
            for other_id, other_pos in self.agent_positions.items():
                if other_id != agent_id and pos == other_pos:
                    return False
        dx = pos[0] - current_pos[0]
        dy = pos[1] - current_pos[1]
        if abs(dx) == 1 and abs(dy) == 1:
            adj1 = (current_pos[0] + dx, current_pos[1])
            adj2 = (current_pos[0], current_pos[1] + dy)
            adj1_is_obstacle = not self.is_in_bounds(adj1) or self.obstacle_grid[adj1[0], adj1[1]] == 1.0
            adj2_is_obstacle = not self.is_in_bounds(adj2) or self.obstacle_grid[adj2[0], adj2[1]] == 1.0
            if adj1_is_obstacle and adj2_is_obstacle:
                return False
        return True

    def is_in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def perform_action(self, agent_id: int, action: Tuple[int, int]):
        """Execute a discrete action for a single agent and return its outcome."""
        if agent_id >= len(self.agents) or agent_id not in self.agent_positions:
            return RobotState((0, 0), 0.0), self.invalid_move_penalty, False
        current_robot_state = self.agents[agent_id].state
        old_pos = current_robot_state.position
        old_orientation = current_robot_state.orientation
        target_pos = (old_pos[0] + action[0], old_pos[1] + action[1])
        is_move_valid = self.is_valid_position(target_pos, agent_id, old_pos, check_other_agents=True)
        new_pos = old_pos
        new_orientation = old_orientation
        reward = 0.0
        coverage_increase_area = 0.0
        if is_move_valid:
            new_pos = target_pos
            if action != (0, 0):
                new_orientation = math.atan2(action[1], action[0])
            new_robot_state = RobotState(position=new_pos, orientation=new_orientation)
            self.agents[agent_id].update_state(new_robot_state)
            self.agent_positions[agent_id] = new_pos
            coverage_before = self.calculate_coverage_area()
            self._update_coverage(agent_id)
            coverage_after = self.calculate_coverage_area()
            coverage_increase_area = max(0, coverage_after - coverage_before)
            reward_coverage = self.gamma_coverage * coverage_increase_area
            orientation_change = min(abs(new_orientation - old_orientation), 2 * math.pi - abs(new_orientation - old_orientation))
            orientation_penalty = self.orientation_cost_factor * (orientation_change / math.pi)
            reward_step = self.step_penalty
            reward = reward_coverage - orientation_penalty + reward_step
        else:
            new_robot_state = current_robot_state
            reward = self.invalid_move_penalty
        done = self.check_coverage_complete()
        return new_robot_state, reward, done

    def _update_coverage(self, agent_id: int):
        """Trigger raycasting coverage update for the agent AND update agent's local map."""
        if not (0 <= agent_id < len(self.agents)):
            return
        agent = self.agents[agent_id]
        agent_state = agent.state
        updated_nodes = self.raycast_coverage_update(agent_state)
        if updated_nodes:
            for node in updated_nodes:
                if self.world_state.graph.has_node(node):
                    data = self.world_state.graph.nodes[node]
                    pc = data.get('pc', 0.0)
                    ntype = data.get('type', 'free')
                    agent.update_local_map(node, pc, ntype)

    def raycast_coverage_update(self, agent_state: RobotState):
        """Update global world_state coverage based on agent's sensor FOV.

        CRITICAL POMDP FIX: Records discovered obstacles in world_state.graph.
        """
        updated_nodes = set()
        center = np.array(agent_state.position, dtype=float)
        curr_cell_rounded = tuple(np.round(center).astype(int))
        if self.world_state.graph.has_node(curr_cell_rounded):
            current_pc = self.world_state.graph.nodes[curr_cell_rounded].get('pc', 0.0)
            if 1.0 > current_pc:
                self.world_state.graph.nodes[curr_cell_rounded]['pc'] = 1.0
                updated_nodes.add(curr_cell_rounded)
            if self.world_state.graph.nodes[curr_cell_rounded].get('type') != 'covered':
                if 1.0 >= self.coverage_threshold:
                    self.world_state.graph.nodes[curr_cell_rounded]['type'] = 'covered'
        immediate_distance_threshold = 1.0
        half_fov = self.fov_radians / 2
        agent_orientation = agent_state.orientation % (2 * math.pi)
        start_angle = agent_orientation - half_fov
        end_angle = agent_orientation + half_fov
        step_size = 0.5
        max_range = float(self.sensor_range)
        num_rays = 32
        for i in range(num_rays):
            fraction = 0.5 if num_rays == 1 else i / (num_rays - 1)
            angle = start_angle + (fraction * self.fov_radians)
            ray_dist = 0.0
            while ray_dist <= max_range:
                ray_dist += step_size
                offset = np.array([math.cos(angle), math.sin(angle)]) * ray_dist
                ray_pos = center + offset
                pos_rounded = tuple(np.round(ray_pos).astype(int))
                if not self.is_in_bounds(pos_rounded):
                    break

                # ✅ POMDP FIX: Check obstacle and RECORD it if discovered
                is_obstacle = (self.obstacle_grid[pos_rounded[0], pos_rounded[1]] == 1.0)
                if is_obstacle:
                    # Mark obstacle in world graph (discovered via sensor)
                    if self.world_state.graph.has_node(pos_rounded):
                        current_type = self.world_state.graph.nodes[pos_rounded].get('type', 'free')
                        if current_type != 'occupied':
                            self.world_state.graph.nodes[pos_rounded]['type'] = 'occupied'
                            self.world_state.graph.nodes[pos_rounded]['pc'] = 0.0  # Obstacles not coverable
                            updated_nodes.add(pos_rounded)
                    break  # Ray blocked by obstacle

                # Free space - update coverage
                if self.world_state.graph.has_node(pos_rounded):
                    new_pc = 1.0 / (1.0 + np.exp(self.coverage_k * (ray_dist - self.coverage_r0)))
                    new_pc = np.clip(new_pc, 0.0, 1.0)
                    current_pc = self.world_state.graph.nodes[pos_rounded].get('pc', 0.0)
                    if new_pc > current_pc:
                        self.world_state.graph.nodes[pos_rounded]['pc'] = new_pc
                        updated_nodes.add(pos_rounded)
                        if new_pc >= self.coverage_threshold:
                            self.world_state.graph.nodes[pos_rounded]['type'] = 'covered'
                else:
                    break
        return updated_nodes

    def calculate_coverage_area(self):
        """Calculate sum of 'pc' values over all nodes existing in the graph (free/covered)."""
        if not self.world_state or not self.world_state.graph:
            return 0.0
        return sum(data.get('pc', 0.0) for node, data in self.world_state.graph.nodes(data=True))

    def calculate_coverage_percentage(self):
        """Calculate percentage of traversable nodes considered covered."""
        if not self.world_state or not self.world_state.graph:
            return 0.0
        total_traversable_nodes = self.world_state.graph.number_of_nodes()
        if total_traversable_nodes == 0:
            return 100.0
        covered_count = sum(1 for node, data in self.world_state.graph.nodes(data=True)
                           if data.get('type') == 'covered' or data.get('pc', 0.0) >= self.coverage_threshold)
        return (covered_count / total_traversable_nodes) * 100.0

    def check_coverage_complete(self):
        """Check if coverage percentage meets the completion threshold."""
        coverage_percentage = self.calculate_coverage_percentage()
        is_complete = coverage_percentage >= self.completion_threshold_perc
        if is_complete and not self.metrics.final_coverage_reached:
            self.metrics.final_coverage_reached = True
            self.metrics.steps_to_threshold = len(self.metrics.total_coverage)
        return is_complete

    def _resolve_conflicts(self, actions):
        """Resolve conflicts: Simple ID-based priority."""
        resolved_actions = actions.copy()
        target_positions = defaultdict(list)
        for agent_id, action in actions.items():
            if agent_id in self.agent_positions:
                current_pos = self.agent_positions[agent_id]
                target_pos = (current_pos[0] + action[0], current_pos[1] + action[1])
                if target_pos != current_pos:
                    if self.is_valid_position(target_pos, agent_id, current_pos, check_other_agents=False):
                        target_positions[target_pos].append(agent_id)
                    else:
                        resolved_actions[agent_id] = (0, 0)
            else:
                resolved_actions[agent_id] = (0, 0)
        for target_pos, competing_agents in target_positions.items():
            if len(competing_agents) > 1:
                competing_agents.sort()
                winner_agent_id = competing_agents[0]
                for agent_id in competing_agents:
                    if agent_id != winner_agent_id:
                        resolved_actions[agent_id] = (0, 0)
        return resolved_actions

    def execute_communications(self):
        """Execute communication between agents within range."""
        comm_events_count = 0
        agents_communicated_this_step = set()
        if len(self.agents) < 2:
            self.metrics.communication_events.append(0)
            return 0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if i < len(self.agents) and j < len(self.agents):
                    agent_i, agent_j = self.agents[i], self.agents[j]
                    if agent_i.communicate(agent_j, self.env_time):
                        pair = tuple(sorted((i, j)))
                        if pair not in agents_communicated_this_step:
                            comm_events_count += 1
                            agents_communicated_this_step.add(pair)
        self.metrics.communication_events.append(comm_events_count)
        return comm_events_count

    def step(self, actions: Dict[int, Tuple[int, int]]):
        """Execute one environment step: resolve conflicts, perform actions, update state."""
        step_start_time = time.time()
        global_state = self.get_global_state_tensor()
        local_states = {i: self.agents[i].get_state_tensor() for i in range(self.num_agents)}
        self.previous_global_coverage = self.calculate_coverage_area()
        self.current_rewards = {i: 0.0 for i in range(self.num_agents)}

        for agent_id in range(self.num_agents):
            if agent_id not in actions:
                actions[agent_id] = (0, 0)
        resolved_actions = self._resolve_conflicts(actions)

        results = {}
        episode_done_flag = False
        individual_rewards = {}
        dones = {}
        next_state_tensors = {}
        agent_action_indices = {}

        for agent_id, action in resolved_actions.items():
            if agent_id < len(self.agents):
                new_robot_state, reward, done = self.perform_action(agent_id, action)
                next_state_tensors[agent_id] = self.agents[agent_id].get_state_tensor()
                individual_rewards[agent_id] = reward
                dones[agent_id] = done
                self.current_rewards[agent_id] = reward
                self.metrics.agent_rewards[agent_id].append(reward)
                if done:
                    episode_done_flag = True
                try:
                    agent_action_indices[agent_id] = self.agents[agent_id].actions.index(action)
                except ValueError:
                    agent_action_indices[agent_id] = self.agents[agent_id].actions.index((0, 0))
            else:
                dummy_state = (torch.zeros(self.input_channels, self.grid_size, self.grid_size, device=self.device),
                              torch.zeros(self.feature_dim, device=self.device))
                next_state_tensors[agent_id] = dummy_state
                individual_rewards[agent_id] = 0.0
                dones[agent_id] = False
                agent_action_indices[agent_id] = 4
                results[agent_id] = (dummy_state[0], dummy_state[1], 0.0, False, {'error': 'invalid agent_id'})

        next_global_state = self.get_global_state_tensor()
        next_local_states = {i: next_state_tensors[i] for i in range(self.num_agents)}
        global_done = any(dones.values()) or self.check_coverage_complete()

        self.current_global_coverage = self.calculate_coverage_area()
        self.global_coverage_increase = self.current_global_coverage - self.previous_global_coverage
        team_reward = self.global_coverage_increase

        self.memory.push(
            global_state, local_states, agent_action_indices, team_reward,
            next_global_state, next_local_states, global_done
        )

        loss = self.optimize_qmix()
        if loss is not None:
            self.metrics.qmix_loss.append(loss)
            if self.writer:
                self.writer.add_scalar('Loss/QMIX_Loss', loss, self.global_optimization_steps)

        self.update_target_networks()
        self.execute_communications()
        self.metrics.total_coverage.append(self.current_global_coverage)
        self.metrics.coverage_rate.append(self.global_coverage_increase)
        step_time = time.time() - step_start_time
        self.metrics.execution_times.append(step_time)
        self.env_time += step_time

        for agent_id in resolved_actions.keys():
            if agent_id < len(self.agents):
                results[agent_id] = (next_state_tensors[agent_id][0], next_state_tensors[agent_id][1],
                                   individual_rewards[agent_id], episode_done_flag, {})

        return results

    def optimize_qmix(self):
        """Perform one step of optimization for QMIX."""
        if len(self.memory) < self.batch_size:
            return None

        self.global_optimization_steps += 1
        transitions = self.memory.sample(self.batch_size)
        batch_global_state_cpu, batch_local_states_cpu, batch_actions_indices, batch_rewards, batch_next_global_cpu, batch_next_locals_cpu, batch_dones = zip(*transitions)

        batch_global_state = torch.stack(batch_global_state_cpu).to(self.device)
        batch_next_global = torch.stack(batch_next_global_cpu).to(self.device)

        agent_q_values = []
        agent_target_max_q = []

        rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float, device=self.device).unsqueeze(-1)
        dones_tensor = torch.tensor(batch_dones, dtype=torch.float, device=self.device).unsqueeze(-1)

        for i in range(self.num_agents):
            agent = self.agents[i]
            local_states_grid = torch.stack([s[i][0] for s in batch_local_states_cpu]).to(self.device)
            local_states_feat = torch.stack([s[i][1] for s in batch_local_states_cpu]).to(self.device)
            actions_idx_list = [batch_actions_indices[b].get(i, 4) for b in range(self.batch_size)]
            actions_idx = torch.tensor(actions_idx_list, dtype=torch.long, device=self.device).unsqueeze(-1)

            next_local_states_grid = torch.stack([s[i][0] for s in batch_next_locals_cpu]).to(self.device)
            next_local_states_feat = torch.stack([s[i][1] for s in batch_next_locals_cpu]).to(self.device)

            current_agent_q_all = agent.policy_net(local_states_grid, local_states_feat)
            current_agent_q = current_agent_q_all.gather(1, actions_idx)
            agent_q_values.append(current_agent_q)

            with torch.no_grad():
                target_agent_q_all = agent.target_net(next_local_states_grid, next_local_states_feat)
                target_agent_max_q_val = target_agent_q_all.max(1)[0].unsqueeze(-1)
            agent_target_max_q.append(target_agent_max_q_val)

        agent_q_values_stacked = torch.cat(agent_q_values, dim=1)
        agent_target_max_q_stacked = torch.cat(agent_target_max_q, dim=1)

        q_tot = self.mixer(agent_q_values_stacked, batch_global_state)
        with torch.no_grad():
            target_q_tot = self.target_mixer(agent_target_max_q_stacked, batch_next_global)
            td_target = rewards_tensor + self.gamma * (1 - dones_tensor) * target_q_tot

        loss = F.mse_loss(q_tot, td_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        all_params = list(self.mixer.parameters())
        for agent in self.agents:
            all_params.extend(list(agent.policy_net.parameters()))
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        self.optimizer.step()

        self.metrics.qmix_loss.append(loss.item())
        self.metrics.q_tot_values.append(q_tot.mean().item())

        return loss.item()

    def update_target_networks(self):
        """Update the target networks (agents and mixer)."""
        if self.use_soft_update:
            for agent in self.agents:
                for target_param, policy_param in zip(agent.target_net.parameters(), agent.policy_net.parameters()):
                    target_param.data.copy_(self.soft_update_tau * policy_param.data + (1.0 - self.soft_update_tau) * target_param.data)
            for target_param, mixer_param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(self.soft_update_tau * mixer_param.data + (1.0 - self.soft_update_tau) * target_param.data)
        else:
            if self.global_optimization_steps % self.target_update_freq == 0:
                for agent in self.agents:
                    agent.target_net.load_state_dict(agent.policy_net.state_dict())
                self.target_mixer.load_state_dict(self.mixer.state_dict())

    def grid_to_matrix(self, graph=None):
        """Convert graph coverage (global or local) to matrix for visualization."""
        target_graph = graph if graph is not None else self.world_state.graph
        matrix = np.full((self.grid_size, self.grid_size), -1.0, dtype=np.float32)
        if target_graph:
            for node, data in target_graph.nodes(data=True):
                r, c = node
                if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                    matrix[r, c] = data.get('pc', 0.0)
        return matrix

    def visualize_coverage(self, trajectories, episode=None, coverage_matrix=None, title_suffix=""):
        """Plot coverage map and agent trajectories."""
        visualize_coverage_plot(
            self, trajectories, episode, coverage_matrix, title_suffix
        )

    def visualize_agent_local_map(self, agent_id, episode=None):
        """Visualizes the final local map perspective of a specific agent."""
        visualize_agent_local_map_plot(self, agent_id, episode)

    def visualize_agent_local_timesteps(self, agent_id, local_map_history, position_history,
                                       orientation_history, episode_label, save_path=None):
        """Animates an agent's local map evolution with position/orientation."""
        visualize_agent_local_timesteps_animation(
            self, agent_id, local_map_history, position_history,
            orientation_history, episode_label, save_path
        )

    def visualize_evaluation_timesteps(self, coverage_history, episode_label, save_path=None):
        """Generates an animation showing global coverage evolution."""
        visualize_evaluation_timesteps_animation(
            self, coverage_history, episode_label, save_path
        )

    def visualize_learning_metrics(self):
        """Visualize learning metrics dashboard."""
        visualize_learning_metrics_dashboard(self)

    def save_models(self, episode, model_dir="./qmix_models"):
        """Saves agent networks and mixer networks."""
        ensure_dir(model_dir)
        state = {
            'episode': episode,
            'mixer_state_dict': self.mixer.state_dict(),
            'target_mixer_state_dict': self.target_mixer.state_dict(),
            'agents_state_dict': [agent.save_agent_state() for agent in self.agents],
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        filename = f"qmix_checkpoint_ep{episode}.pt"
        final_path = os.path.join(model_dir, filename)
        try:
            torch.save(state, final_path)
        except Exception as e:
            print(f"Error saving QMIX models: {e}")

    def load_models(self, episode_tag, model_dir="./qmix_models"):
        """Loads agent networks and mixer networks."""
        filename = f"qmix_checkpoint_ep{episode_tag}.pt"
        model_path = os.path.join(model_dir, filename)
        models_loaded = False
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.mixer.load_state_dict(checkpoint['mixer_state_dict'])
                self.target_mixer.load_state_dict(checkpoint['target_mixer_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                agents_state_dict = checkpoint['agents_state_dict']
                if len(agents_state_dict) == self.num_agents:
                    for i, agent in enumerate(self.agents):
                        agent.load_agent_state(agents_state_dict[i])
                    print(f"QMIX models loaded from {model_path} (Episode {checkpoint.get('episode', episode_tag)})")
                    models_loaded = True
                else:
                    print(f"Error loading: Agent count mismatch.")
                self.target_mixer.eval()
            except Exception as e:
                print(f"Error loading QMIX checkpoint from {model_path}: {e}")
        else:
            print(f"Warning: No QMIX checkpoint found at {model_path}")
        return models_loaded
