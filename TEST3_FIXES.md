# Hyperparameter Tuning for Test 3

## Problem: Loss exploding in obstacle environments

### Quick Fixes to Try

1. **Lower Learning Rate**
```python
lr = 0.0001  # Was 0.0005, too aggressive
```

2. **Increase Batch Size**
```python
batch_size = 128  # Was 64, more stable gradients
```

3. **Gradient Clipping** (already at 1.0, try stricter)
```python
torch.nn.utils.clip_grad_norm_(all_params, 0.5)  # Was 1.0
```

4. **Reward Scaling**
```python
gamma_coverage = 10.0  # Was 25.0, reduce reward magnitude
```

5. **Target Network Updates** (slower = more stable)
```python
target_update_freq = 500  # Was 200
# OR increase soft update tau
soft_update_tau = 0.001  # Was 0.005
```

6. **Exploration Decay** (explore longer in complex envs)
```python
epsilon_decay = 0.995  # Was 0.99, slower decay
epsilon_end = 0.1  # Was 0.05, more exploration
```

### Test Configuration for Stability

```python
env = MARL_QMIX_Environment(
    # ... other params ...
    lr=0.0001,  # Lower LR
    batch_size=128,  # Larger batch
    gamma_coverage=10.0,  # Lower rewards
    target_update_freq=500,  # Slower target updates
    agent_config={
        'epsilon_start': 1.0,
        'epsilon_end': 0.1,
        'epsilon_decay': 0.995,  # Slower decay
    }
)
```

### Why Loss Explodes

1. **Q-values grow unbounded** in obstacle scenarios
2. **Sparse positive rewards** → network overfits to rare successes
3. **Long episodes** → more steps → more error accumulation
4. **Target network divergence** → TD error grows

### Alternative: Reward Shaping

Add intermediate rewards to stabilize learning:
```python
# In perform_action(), add:
if coverage_increase_area > 0:
    reward_coverage = self.gamma_coverage * coverage_increase_area
else:
    # Small reward for exploring new areas
    reward_coverage = 0.1 if new_pos != old_pos else 0.0
```

---

**For publication**: 2/3 passing is acceptable. Document Test 3 as "future work" requiring reward shaping or hierarchical RL.
