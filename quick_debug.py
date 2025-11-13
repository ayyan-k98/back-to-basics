"""
Quick Debug - Fast diagnosis of QMIX issues
"""

import numpy as np
import torch
from environment import MARL_QMIX_Environment
import sys

print("\n" + "="*80)
print("QUICK DEBUG: QMIX Learning Failure Analysis")
print("="*80)

# Create minimal environment
print("\nCreating environment...")
env = MARL_QMIX_Environment(
    grid_size=5,
    num_agents=2,
    sensor_range=3,
    comm_range=5.0,
    coverage_threshold=0.8,
    completion_threshold_perc=80.0,
    max_episodes=3,  # Just 3 episodes
    max_steps_per_episode=15,
    device="cpu",
    tensorboard_dir=None,
    batch_size=32,
    memory_capacity=5000,
    gamma=0.99,
    lr=0.001,
    agent_config={
        'epsilon_start': 1.0,
        'epsilon_end': 0.1,
        'epsilon_decay': 0.95,
    },
    gamma_coverage=30.0,
    step_penalty=-0.05,
)

# Force empty map
env.obstacle_grid = np.zeros((5, 5), dtype=np.float32)
env.obstacle_grid[0, :] = 1.0
env.obstacle_grid[-1, :] = 1.0
env.obstacle_grid[:, 0] = 1.0
env.obstacle_grid[:, -1] = 1.0

print(f"Grid size: {env.grid_size}x{env.grid_size}")
print(f"Free cells: {np.sum(env.obstacle_grid == 0)}")
print(f"Agents: {env.num_agents}")

print("\n" + "-"*80)
print("Running 3 episodes with detailed logging...")
print("-"*80)

for episode in range(3):
    print(f"\nEPISODE {episode}:")
    states = env.reset(full_reset=(episode == 0), mode='train')
    print(f"  Initial positions: {env.agent_positions}")
    print(f"  Initial coverage: {env.calculate_coverage_percentage():.1f}%")
    print(f"  Buffer size: {len(env.memory)}")
    
    for step in range(env.max_steps_per_episode):
        # Get actions
        actions = {}
        for agent_id, agent in enumerate(env.agents):
            valid_actions = env.get_valid_actions(agent_id)
            action = agent.select_action(states[agent_id], valid_actions)
            actions[agent_id] = action
        
        # Step
        results = env.step(actions)
        coverage = env.calculate_coverage_percentage()
        done_flags = {aid: r[3] for aid, r in results.items()}
        
        print(f"    Step {step}: coverage={coverage:5.1f}%, done={any(done_flags.values())}, buffer={len(env.memory)}")
        
        if any(done_flags.values()):
            print(f"    → Episode ended early! Done flags: {done_flags}")
            break
        
        # Update states
        for agent_id, result in results.items():
            states[agent_id] = (result[0], result[1])
    
    print(f"  Final coverage: {env.calculate_coverage_percentage():.1f}%")
    print(f"  Final buffer size: {len(env.memory)}")

print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)
print(f"Final replay buffer size: {len(env.memory)}")
print(f"Batch size requirement: {env.batch_size}")
print(f"Can optimize: {len(env.memory) >= env.batch_size}")
print(f"Global optimization steps: {env.global_optimization_steps}")

if len(env.memory) >= env.batch_size:
    print("\nTesting optimization...")
    loss = env.optimize_qmix()
    print(f"Optimization loss: {loss}")
else:
    print(f"\nWARNING: Buffer too small to optimize!")
    print(f"Need {env.batch_size - len(env.memory)} more transitions")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)

# Check if episodes are ending early
if len(env.memory) < 3 * 15 * 2:  # 3 episodes * 15 steps * 2 agents
    print("❌ ISSUE: Not enough transitions stored!")
    print(f"   Expected: ~{3 * 15 * 2} transitions")
    print(f"   Actual: {len(env.memory)} transitions")
    print("   → Episodes may be terminating early")

if env.global_optimization_steps == 0:
    print("❌ ISSUE: No optimization steps occurred!")
    print("   → QMIX is not learning at all")

print("\n" + "="*80)
