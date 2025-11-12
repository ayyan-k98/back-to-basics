# Implementation Roadmap: Publication-Quality QMIX Coverage System

## 🎯 Goal: Transform current 4/10 → 8+/10 publication-ready system

---

## Phase 1: Grid-Size Agnostic Architecture (CRITICAL) 🔴
**Timeline**: Week 1-2
**Priority**: HIGHEST
**Current**: 0/10 → **Target**: 8/10

### Why This is Critical
Your network architectures are hardcoded to grid size:
```python
# environment.py:232
global_state_np = np.concatenate([flat_coverage, flat_obstacles, flat_agent_pos, flat_orientations])
# This changes size with grid_size! 20x20 = 800, 30x30 = 1800
```

**Problem**: Train on 20x20, cannot deploy on 50x50 without retraining completely.

### Solution Approaches

#### Option A: Local Observation Windows (RECOMMENDED - Easiest)
**Complexity**: Medium
**Expected Performance**: Good
**Implementation Time**: 3-4 days

1. **Create local observation extractor**:
```python
# In agent.py
def get_local_observation(self, window_size=7):
    """Extract fixed-size local window around agent."""
    r, c = self.state.position
    half_w = window_size // 2

    # Extract local coverage
    local_coverage = np.zeros((window_size, window_size))
    local_obstacles = np.zeros((window_size, window_size))

    for dr in range(-half_w, half_w + 1):
        for dc in range(-half_w, half_w + 1):
            world_r, world_c = r + dr, c + dc
            local_r, local_c = dr + half_w, dc + half_w

            if self.env.is_in_bounds((world_r, world_c)):
                if self.local_map.has_node((world_r, world_c)):
                    data = self.local_map.nodes[(world_r, world_c)]
                    local_coverage[local_r, local_c] = data.get('pc', 0.0)
                    local_obstacles[local_r, local_c] = 0.0
                else:
                    local_obstacles[local_r, local_c] = 1.0
            else:
                local_obstacles[local_r, local_c] = 1.0  # Out of bounds = wall

    # Add relative position encoding
    rel_pos_r = r / self.grid_size  # Normalized position
    rel_pos_c = c / self.grid_size

    return local_coverage, local_obstacles, rel_pos_r, rel_pos_c
```

2. **Modify network to accept local observations**:
```python
# In networks.py
class GridAgnosticDuelingDQN(nn.Module):
    def __init__(self, window_size=7, num_actions=9):
        super().__init__()
        self.window_size = window_size

        # Fixed-size convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),  # 7x7 -> 7x7
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),  # 7x7 -> 7x7
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),  # 7x7 -> 7x7
            nn.ReLU()
        )

        conv_out_size = 64 * window_size * window_size

        # Extra features: orientation (2) + relative position (2)
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size + 4, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size + 4, 256), nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, local_grids, extra_features):
        # local_grids: [batch, 2, 7, 7]
        # extra_features: [batch, 4] (sin_ori, cos_ori, rel_r, rel_c)
        batch_size = local_grids.size(0)

        conv_out = self.conv(local_grids)
        conv_flat = conv_out.view(batch_size, -1)

        combined = torch.cat([conv_flat, extra_features], dim=1)

        value = self.value_stream(combined)
        advantage = self.advantage_stream(combined)

        return value + advantage - advantage.mean(dim=1, keepdim=True)
```

3. **Update global state to be grid-agnostic**:
```python
# Option 1: Summary statistics (size-invariant)
def get_global_state_tensor(self):
    """Grid-agnostic global state using summary statistics."""
    # Coverage statistics
    coverage_values = [data.get('pc', 0.0) for _, data in self.world_state.graph.nodes(data=True)]
    coverage_mean = np.mean(coverage_values)
    coverage_std = np.std(coverage_values)
    coverage_max = np.max(coverage_values)
    coverage_min = np.min(coverage_values)

    # Agent positions (normalized)
    agent_positions_flat = []
    for agent in self.agents:
        r, c = agent.state.position
        agent_positions_flat.extend([r / self.grid_size, c / self.grid_size])

    # Agent orientations
    agent_orientations = []
    for agent in self.agents:
        agent_orientations.extend([
            math.sin(agent.state.orientation),
            math.cos(agent.state.orientation)
        ])

    # Combine into fixed-size vector
    global_features = [
        coverage_mean, coverage_std, coverage_max, coverage_min,
        *agent_positions_flat,
        *agent_orientations
    ]

    return torch.FloatTensor(global_features).to(self.device)
```

**Testing Protocol**:
```python
# Test script to verify grid-agnostic property
def test_grid_agnostic():
    grid_sizes = [10, 15, 20, 25, 30, 40, 50]

    # Train on medium size
    env_20 = create_env(grid_size=20)
    train(env_20, episodes=100)
    env_20.save_models(100, "models_20x20")

    # Test on all sizes
    results = {}
    for size in grid_sizes:
        env_test = create_env(grid_size=size)
        env_test.load_models(100, "models_20x20")  # Load 20x20 model
        metrics = evaluate(env_test, episodes=20)
        results[size] = metrics

    # Plot transfer performance
    plot_transfer_performance(results)
```

#### Option B: Adaptive Pooling (Quick Fix)
**Complexity**: Low
**Expected Performance**: Moderate
**Implementation Time**: 1 day

```python
# Add to DuelingConvDQN
self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # Always pool to 7x7

def forward(self, x, features):
    x = self.conv_layers(x)
    x = self.adaptive_pool(x)  # Now always 7x7 regardless of input size
    # ... rest unchanged
```

**Pros**: Quick to implement
**Cons**: May lose spatial information, limited generalization

#### Option C: Graph Neural Networks (Most Robust)
**Complexity**: High
**Expected Performance**: Excellent
**Implementation Time**: 1-2 weeks

Convert grid to graph, use GNN for processing. Naturally scales to any grid size.

**Recommendation**: Start with **Option A** (local windows), fall back to **Option B** if time-constrained.

---

## Phase 2: Overlap Minimization (CORE CONTRIBUTION) 🔴
**Timeline**: Week 3
**Priority**: HIGHEST
**Current**: 1/10 → **Target**: 8/10

### Implementation Steps

#### 2.1: Track Overlap Metrics
```python
# In environment.py
class MARL_QMIX_Environment:
    def __init__(self, ...):
        # ...
        self.cell_visit_counts = np.zeros((grid_size, grid_size), dtype=int)
        self.agent_visit_counts = [
            np.zeros((grid_size, grid_size), dtype=int)
            for _ in range(num_agents)
        ]

    def _update_coverage(self, agent_id):
        # ... existing code ...

        # Track visits
        pos = self.agents[agent_id].state.position
        self.cell_visit_counts[pos] += 1
        self.agent_visit_counts[agent_id][pos] += 1

    def calculate_overlap_metrics(self):
        """Calculate comprehensive overlap metrics."""
        total_cells_visited = np.sum(self.cell_visit_counts > 0)
        total_visits = np.sum(self.cell_visit_counts)

        # Redundancy ratio
        redundancy = (total_visits - total_cells_visited) / max(total_visits, 1)

        # Overlap matrix (how much agents overlap)
        overlap_matrix = np.zeros((self.num_agents, self.num_agents))
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                overlap = np.sum(
                    (self.agent_visit_counts[i] > 0) &
                    (self.agent_visit_counts[j] > 0)
                )
                overlap_matrix[i, j] = overlap
                overlap_matrix[j, i] = overlap

        # Average cells visited multiple times
        multi_visit_cells = np.sum(self.cell_visit_counts > 1)

        return {
            'redundancy_ratio': redundancy,
            'overlap_matrix': overlap_matrix,
            'multi_visit_cells': multi_visit_cells,
            'total_cells_visited': total_cells_visited,
            'avg_visits_per_cell': total_visits / max(total_cells_visited, 1)
        }
```

#### 2.2: Add Overlap Penalty to Reward
```python
# In config.py
REWARD_PARAMS = {
    'gamma_coverage': 20.0,
    'step_penalty': -0.01,
    'orientation_cost_factor': 0.02,
    'invalid_move_penalty': -0.5,
    'overlap_penalty': -0.3,        # NEW
    'novelty_bonus': 0.5,           # NEW
    'diversity_bonus': 0.2,         # NEW
}

# In environment.py - perform_action()
def perform_action(self, agent_id, action):
    # ... existing code ...

    if is_move_valid:
        # ... movement logic ...

        # Check if position was already visited
        if self.cell_visit_counts[new_pos] > 0:
            overlap_penalty = self.overlap_penalty * self.cell_visit_counts[new_pos]
            reward += overlap_penalty
        else:
            novelty_bonus = self.novelty_bonus  # First visit bonus
            reward += novelty_bonus

        # Coverage reward (existing)
        coverage_before = self.calculate_coverage_area()
        self._update_coverage(agent_id)
        coverage_after = self.calculate_coverage_area()
        coverage_increase_area = max(0, coverage_after - coverage_before)
        reward_coverage = self.gamma_coverage * coverage_increase_area

        # ... rest of reward calculation ...
```

#### 2.3: Spatial Diversity Bonus
```python
def calculate_spatial_diversity_bonus(self):
    """Reward agents for being spread out."""
    if len(self.agents) < 2:
        return 0.0

    positions = [agent.state.position for agent in self.agents]

    # Minimum pairwise distance
    min_dist = float('inf')
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(
                np.array(positions[i]) - np.array(positions[j])
            )
            min_dist = min(min_dist, dist)

    # Normalize by grid diagonal
    max_dist = np.sqrt(2 * self.grid_size ** 2)
    normalized_min_dist = min_dist / max_dist

    # Bonus for being far apart
    return self.diversity_bonus * normalized_min_dist
```

#### 2.4: Voronoi Territory Assignment (Advanced)
```python
from scipy.spatial import Voronoi

def assign_territories(self):
    """Assign Voronoi territories to agents."""
    if len(self.agents) < 2:
        return None

    agent_positions = [agent.state.position for agent in self.agents]

    # Create Voronoi diagram
    vor = Voronoi(agent_positions)

    # Assign each cell to nearest agent
    territories = {}
    for r in range(self.grid_size):
        for c in range(self.grid_size):
            if self.world_state.graph.has_node((r, c)):
                # Find nearest agent
                min_dist = float('inf')
                nearest_agent = 0
                for i, pos in enumerate(agent_positions):
                    dist = (r - pos[0])**2 + (c - pos[1])**2
                    if dist < min_dist:
                        min_dist = dist
                        nearest_agent = i
                territories[(r, c)] = nearest_agent

    return territories

# In reward calculation
def territory_bonus(self, agent_id, position):
    """Bonus for covering own territory."""
    territories = self.assign_territories()
    if territories is None:
        return 0.0

    if territories.get(position) == agent_id:
        return 0.1  # Small bonus for covering own territory
    else:
        return -0.05  # Small penalty for covering others' territory
```

---

## Phase 3: Advanced Exploration Strategies 🟡
**Timeline**: Week 4
**Priority**: HIGH
**Current**: 3/10 → **Target**: 7/10

### 3.1: Count-Based Exploration
```python
# In agent.py
class MARLCoverageAgent:
    def __init__(self, ...):
        # ...
        self.state_visit_counts = {}
        self.exploration_bonus_weight = 0.5

    def get_exploration_bonus(self, state):
        """Compute count-based exploration bonus."""
        state_key = self._state_to_key(state)
        count = self.state_visit_counts.get(state_key, 0)

        # Bonus inversely proportional to sqrt(count)
        bonus = self.exploration_bonus_weight / np.sqrt(count + 1)
        return bonus

    def _state_to_key(self, state):
        """Convert state to hashable key."""
        grid, features = state
        # Use position + coverage summary as key
        pos = self.state.position
        coverage_sum = grid[0].sum().item()  # Sum of local coverage
        return (pos, int(coverage_sum * 10))  # Discretize coverage
```

### 3.2: Frontier-Based Exploration
```python
def detect_frontiers(self):
    """Identify frontier cells (boundary of explored area)."""
    frontiers = set()

    for node, data in self.world_state.graph.nodes(data=True):
        if data.get('pc', 0.0) < self.coverage_threshold:
            # Check if adjacent to covered cell
            r, c = node
            neighbors = [
                (r-1, c), (r+1, c), (r, c-1), (r, c+1),
                (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)
            ]

            for neighbor in neighbors:
                if self.world_state.graph.has_node(neighbor):
                    if self.world_state.graph.nodes[neighbor].get('pc', 0.0) >= self.coverage_threshold:
                        frontiers.add(node)
                        break

    return list(frontiers)

def assign_frontiers_to_agents(self):
    """Assign nearest frontier to each agent."""
    frontiers = self.detect_frontiers()
    if not frontiers:
        return {}

    assignments = {}
    available_frontiers = set(frontiers)

    for agent in self.agents:
        if not available_frontiers:
            break

        # Find nearest frontier
        agent_pos = np.array(agent.state.position)
        min_dist = float('inf')
        nearest_frontier = None

        for frontier in available_frontiers:
            dist = np.linalg.norm(agent_pos - np.array(frontier))
            if dist < min_dist:
                min_dist = dist
                nearest_frontier = frontier

        if nearest_frontier:
            assignments[agent.agent_id] = nearest_frontier
            available_frontiers.remove(nearest_frontier)

    return assignments
```

### 3.3: Intrinsic Curiosity Module (ICM) - Advanced
```python
# Create new file: curiosity.py
import torch
import torch.nn as nn

class ForwardModel(nn.Module):
    """Predict next state given current state and action."""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        # action is one-hot encoded
        action_onehot = F.one_hot(action, num_classes=9).float()
        x = torch.cat([state, action_onehot], dim=-1)
        return self.model(x)

class ICM:
    """Intrinsic Curiosity Module."""
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.forward_model = ForwardModel(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.forward_model.parameters(), lr=lr)

    def compute_intrinsic_reward(self, state, action, next_state):
        """Compute curiosity reward as prediction error."""
        predicted_next_state = self.forward_model(state, action)
        prediction_error = F.mse_loss(predicted_next_state, next_state, reduction='none')
        intrinsic_reward = prediction_error.mean().item()
        return intrinsic_reward

    def train_step(self, state, action, next_state):
        """Update forward model."""
        predicted = self.forward_model(state, action)
        loss = F.mse_loss(predicted, next_state)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
```

---

## Phase 4: Robust Metrics & Evaluation 🟡
**Timeline**: Week 5
**Priority**: MEDIUM-HIGH
**Current**: 5/10 → **Target**: 9/10

### 4.1: Comprehensive Metrics Class
```python
# In data_structures.py - enhance CoverageMetrics
@dataclass
class RobustCoverageMetrics:
    # Existing
    total_coverage: List[float] = field(default_factory=list)
    coverage_rate: List[float] = field(default_factory=list)

    # NEW: Overlap metrics
    overlap_ratio: List[float] = field(default_factory=list)
    redundancy_ratio: List[float] = field(default_factory=list)
    multi_visit_percentage: List[float] = field(default_factory=list)

    # NEW: Efficiency metrics
    coverage_efficiency: List[float] = field(default_factory=list)  # coverage / distance
    area_per_step: List[float] = field(default_factory=list)

    # NEW: Convergence metrics
    time_to_threshold: Dict[float, Optional[int]] = field(default_factory=lambda: {
        0.5: None, 0.7: None, 0.8: None, 0.9: None, 0.95: None
    })

    # NEW: Spatial metrics
    spatial_variance: List[float] = field(default_factory=list)
    agent_dispersion: List[float] = field(default_factory=list)
    territory_balance: List[float] = field(default_factory=list)

    # NEW: Exploration metrics
    exploration_entropy: List[float] = field(default_factory=list)
    frontier_count: List[int] = field(default_factory=list)
    unique_cells_visited: List[int] = field(default_factory=list)

    # NEW: Communication metrics
    communication_efficiency: List[float] = field(default_factory=list)
    information_gain_per_comm: List[float] = field(default_factory=list)
```

### 4.2: Enhanced Visualization
```python
# In visualization.py
def visualize_overlap_heatmap(env, episode=None):
    """Visualize visit count heatmap showing overlap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Visit counts
    ax = axes[0]
    im = ax.imshow(env.cell_visit_counts.T, cmap='hot', origin='lower')
    ax.set_title('Cell Visit Counts (Overlap Indicator)')
    plt.colorbar(im, ax=ax, label='# Visits')

    # Redundancy map (visits > 1)
    ax = axes[1]
    redundancy_map = (env.cell_visit_counts > 1).astype(float)
    im = ax.imshow(redundancy_map.T, cmap='Reds', origin='lower', vmin=0, vmax=1)
    ax.set_title('Redundant Coverage (Multiple Visits)')
    plt.colorbar(im, ax=ax, label='Redundant')

    plt.tight_layout()
    plt.show()

def plot_metric_comparison(results_dict, metric_name):
    """Compare metric across different methods/configurations."""
    plt.figure(figsize=(10, 6))

    for method_name, metrics in results_dict.items():
        values = metrics.get(metric_name, [])
        if values:
            plt.plot(values, label=method_name, alpha=0.7)

    plt.xlabel('Step')
    plt.ylabel(metric_name)
    plt.title(f'Comparison: {metric_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

---

## Phase 5: Baseline Implementations 🟡
**Timeline**: Week 6
**Priority**: MEDIUM
**Current**: 0/10 → **Target**: 8/10

### 5.1: Random Policy Baseline
```python
# Create baselines.py
class RandomCoverageAgent:
    """Random action selection baseline."""
    def select_action(self, state, valid_actions):
        return random.choice(valid_actions)

def evaluate_random_baseline(env, num_episodes=20):
    """Evaluate random policy."""
    # Replace agent action selection with random
    # ... implementation
```

### 5.2: Greedy Coverage Baseline
```python
class GreedyCoverageAgent:
    """Always move towards nearest uncovered cell."""
    def select_action(self, state, valid_actions):
        # Find nearest uncovered cell using BFS
        target = self.find_nearest_uncovered()

        # Select action that moves closer to target
        best_action = self.get_action_towards(target, valid_actions)
        return best_action
```

### 5.3: Independent DQN (No QMIX)
```python
# Modify environment to support independent training
class IndependentDQNEnvironment(MARL_QMIX_Environment):
    """Same env but agents train independently."""
    def optimize_independent(self):
        """Each agent optimizes its own network independently."""
        for agent in self.agents:
            if len(agent.memory) < self.batch_size:
                continue

            # Sample from agent's own replay buffer
            batch = agent.memory.sample(self.batch_size)

            # Standard DQN update for this agent
            # ... implementation
```

### 5.4: VDN Baseline
```python
# Implement Value Decomposition Networks as baseline
class VDNMixer(nn.Module):
    """Simple sum instead of QMIX's complex mixing."""
    def forward(self, agent_qs, states):
        # VDN: Q_tot = sum of Q_i
        return agent_qs.sum(dim=1, keepdim=True)
```

---

## Phase 6: Scaling Studies 🟢
**Timeline**: Week 7-8
**Priority**: MEDIUM
**Current**: 0/10 → **Target**: 7/10

### 6.1: Grid Size Scaling Experiment
```python
def run_scaling_experiments():
    """Test performance across different grid sizes."""
    grid_sizes = [10, 15, 20, 25, 30, 40, 50]
    agent_counts = [2, 4, 6, 8]

    results = {}

    for grid_size in grid_sizes:
        for num_agents in agent_counts:
            key = f"{grid_size}x{grid_size}_{num_agents}agents"

            env = create_env(grid_size=grid_size, num_agents=num_agents)

            # Train
            train_metrics = train(env, episodes=100)

            # Evaluate
            eval_metrics = evaluate(env, episodes=20)

            results[key] = {
                'train': train_metrics,
                'eval': eval_metrics,
                'final_coverage': eval_metrics['avg_coverage'],
                'steps_to_90': train_metrics.time_to_threshold[0.9]
            }

    # Plot scaling curves
    plot_scaling_curves(results)
    return results
```

### 6.2: Transfer Learning Experiments
```python
def test_transfer_learning():
    """Train on one size, test on others."""
    # Train on 20x20
    env_train = create_env(grid_size=20, num_agents=4)
    train(env_train, episodes=200)
    env_train.save_models(200, "transfer_20x20")

    # Test on multiple sizes WITHOUT retraining
    test_sizes = [10, 15, 25, 30, 40, 50]
    transfer_results = {}

    for size in test_sizes:
        env_test = create_env(grid_size=size, num_agents=4)
        env_test.load_models(200, "transfer_20x20")

        # Evaluate without training
        metrics = evaluate(env_test, episodes=20)
        transfer_results[size] = metrics

    # Measure transfer gap
    plot_transfer_gap(transfer_results)
```

---

## Critical Success Metrics

### For Publication Acceptance:

1. **Grid-Size Agnostic**: ✅
   - Train 20x20, test on 10-50x50
   - <20% performance drop on unseen sizes

2. **Overlap Minimization**: ✅
   - Overlap ratio < 15%
   - 2x better than independent DQN

3. **Coverage Efficiency**: ✅
   - 90% coverage in <100 steps (20x20, 4 agents)
   - Better than all baselines

4. **Scalability**: ✅
   - Linear or sub-linear scaling with grid size
   - Handles 10-100x100 grids

5. **Statistical Significance**: ✅
   - 10+ seeds per experiment
   - p < 0.05 vs. baselines

---

## Timeline Summary

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1-2 | Grid-agnostic architecture | Working local observation model |
| 3 | Overlap minimization | Overlap metrics + penalties |
| 4 | Advanced exploration | Count-based + frontiers |
| 5 | Robust metrics | 15+ metrics implemented |
| 6 | Baselines | 4 baselines implemented |
| 7-8 | Scaling studies | Experiments on 5+ grid sizes |
| 9-10 | Writing & analysis | Paper draft + theory |

**Total**: ~10 weeks to publication-ready system

---

## Next Immediate Steps (This Week)

1. **Day 1-2**: Implement local observation windows
2. **Day 3**: Add overlap tracking
3. **Day 4**: Implement overlap penalty
4. **Day 5**: Test on multiple grid sizes
5. **Weekend**: Run experiments, collect data

**Goal**: By end of Week 1, have grid-agnostic system with basic overlap minimization.
