# QMIX Multi-Agent Coverage System

A modular implementation of QMIX (Q-Mixing Network) for multi-agent reinforcement learning applied to coverage tasks. This system uses Dueling DQN as agent networks and QMIX for value function decomposition.

---

## ⚠️ CURRENT STATUS: VALIDATION PHASE

**What We Have:**
- ✅ Modular codebase
- ✅ QMIX implementation
- ✅ **TRUE POMDP** (partial observability verified)
- ✅ Validation test suite

**What We DON'T Have:**
- ❌ **Proof that QMIX learns** (validation pending)
- ❌ Baseline comparisons
- ❌ Performance benchmarks

**Next Step:** Run validation tests to verify QMIX works on toy problems

```bash
python validate.py  # 30-60 min runtime
```

**Why validation first?** See [VALIDATION_FIRST.md](VALIDATION_FIRST.md) for philosophy.

**DO NOT add features until validation passes!**

---

## Project Structure

```
back-to-basics/
├── main.py                  # Entry point for training and evaluation
├── config.py               # Configuration parameters
├── environment.py          # MARL QMIX environment
├── agent.py               # Individual agent implementation
├── networks.py            # Neural network architectures (DuelingConvDQN, QMixNetwork)
├── memory.py              # Replay buffer for QMIX
├── data_structures.py     # Data classes and structures
├── utils.py               # Utility functions (map generation, helpers)
├── visualization.py       # Visualization functions
├── train.py              # Training and evaluation logic
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Features

- **QMIX Architecture**: Centralized training with decentralized execution
- **Dueling DQN**: Each agent uses Dueling DQN for individual Q-value estimation
- **Multiple Map Types**: Training on room, cave, and random maps
- **Communication**: Agents can exchange local map information
- **Visualization**: Coverage maps, learning metrics, agent trajectories, and animations
- **TensorBoard**: Training metrics logging

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For animations (optional):
```bash
# Install ffmpeg for saving animations
sudo apt-get install ffmpeg  # Ubuntu/Debian
# or
brew install ffmpeg  # macOS
```

## Usage

### Basic Training and Evaluation

```bash
python main.py
```

### Configuration

Edit `config.py` to customize:

- **Grid and Agent Settings**: Grid size, number of agents, sensor range, etc.
- **Training Parameters**: Number of episodes, batch size, learning rate, etc.
- **Reward Parameters**: Coverage reward, step penalty, etc.
- **Visualization Flags**: Enable/disable various visualizations

### Key Configuration Parameters

```python
# Grid and Agent Configuration
GRID_SIZE = 20
NUM_AGENTS = 4
SENSOR_RANGE = 5
COMM_RANGE = 8.0

# QMIX Parameters
QMIX_PARAMS = {
    'memory_capacity': 50000,
    'batch_size': 128,
    'gamma': 0.99,
    'lr': 0.0005,
    'mixer_embed_dim': 64,
}

# Agent Parameters
AGENT_HYPERPARAMS = {
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay': 0.999,
    'use_dueling': True,
}
```

## Module Descriptions

### `environment.py`
- `MARL_QMIX_Environment`: Main environment class managing the multi-agent system
- Handles map generation, agent coordination, QMIX optimization, and metrics tracking

### `agent.py`
- `MARLCoverageAgent`: Individual agent with local map knowledge
- Manages exploration-exploitation (epsilon-greedy), communication, and state representation

### `networks.py`
- `DuelingConvDQN`: Agent network with convolutional layers and dueling architecture
- `QMixNetwork`: Mixing network combining individual Q-values into Q_tot

### `memory.py`
- `QMixReplayMemory`: Experience replay buffer for storing joint transitions

### `train.py`
- `train()`: Main training loop with map cycling and periodic saving
- `evaluate()`: Evaluation on random maps with metrics

### `visualization.py`
- Coverage map plotting
- Learning metrics dashboard
- Agent trajectory visualization
- Animation generation for timesteps

### `utils.py`
- Map generation functions (room, cave, random, empty)
- Directory management
- Moving average for smoothing metrics

### `data_structures.py`
- Dataclasses for robot state, world state, metrics, communication events, rooms

## Output

The system generates several outputs:

1. **Models**: Saved to `MODEL_DIR` (default: `./models_coverage_qmix_final`)
2. **TensorBoard Logs**: Saved to `RUN_DIR` (default: `./runs/coverage_qmix_experiment_final`)
3. **Animations**: Saved to `ANIMATION_DIR` (default: `./eval_animations_qmix_final`)

### Viewing TensorBoard Logs

```bash
tensorboard --logdir=./runs/coverage_qmix_experiment_final
```

## Training Process

1. **Initialization**: Environment generates maps and initializes agents
2. **Episode Loop**:
   - Agents select actions using epsilon-greedy
   - Environment resolves conflicts and executes actions
   - Coverage is updated via raycasting
   - Agents communicate to share local maps
   - Experience stored in replay buffer
   - QMIX optimization step
3. **Periodic Saves**: Models saved at specified intervals
4. **Evaluation**: Trained agents evaluated on random maps

## Key Algorithms

- **QMIX**: Value function decomposition for cooperative multi-agent RL
- **Dueling DQN**: Separate value and advantage streams in agent networks
- **Epsilon-Greedy**: Exploration-exploitation balance
- **Raycasting**: Coverage calculation with field-of-view
- **Conflict Resolution**: Priority-based agent coordination

## Customization

### Adding New Map Types

Edit `utils.py` and add a new map generation function, then update `environment.py`:

```python
def generate_custom_map(grid_size):
    # Your custom map generation logic
    return grid

# In environment.py
self.training_map_generators['custom'] = lambda: generate_custom_map(self.grid_size)
self.training_map_order = ['room', 'cave', 'random', 'custom']
```

### Modifying Network Architecture

Edit `networks.py` to change the neural network architecture:

```python
class CustomDQN(nn.Module):
    def __init__(self, ...):
        # Your custom architecture
```

### Adjusting Rewards

Edit `config.py` to modify reward parameters:

```python
REWARD_PARAMS = {
    'gamma_coverage': 20.0,
    'step_penalty': -0.01,
    'orientation_cost_factor': 0.02,
    'invalid_move_penalty': -0.5,
}
```

## Performance Tips

1. **GPU Acceleration**: Set `DEVICE = "cuda"` in `config.py` if CUDA is available
2. **Batch Size**: Increase for better gradient estimates (requires more memory)
3. **Memory Capacity**: Increase for more diverse experience replay
4. **Target Update**: Adjust `target_update_freq` or use soft updates

## Troubleshooting

- **Out of Memory**: Reduce batch size or memory capacity
- **Slow Training**: Enable GPU, reduce grid size, or reduce number of agents
- **Poor Coverage**: Adjust reward parameters, increase training episodes
- **No Improvement**: Check learning rate, ensure proper target network updates

## Citation

If you use this code in your research, please cite:

```
@software{qmix_coverage_2024,
  author = {Your Name},
  title = {QMIX Multi-Agent Coverage System},
  year = {2024},
  url = {https://github.com/yourusername/back-to-basics}
}
```

## License

[Your chosen license]

## References

- QMIX Paper: Rashid et al., "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning", ICML 2018
- Dueling DQN: Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", ICML 2016
