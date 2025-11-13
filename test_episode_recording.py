"""
Minimal test to verify episode metrics recording
"""
import numpy as np
from environment import MARL_QMIX_Environment
from train import train

print("Creating environment...")
env = MARL_QMIX_Environment(
    grid_size=5,
    num_agents=2,
    sensor_range=3,
    comm_range=5.0,
    coverage_threshold=0.8,
    completion_threshold_perc=80.0,
    max_episodes=5,  # Just 5 episodes
    max_steps_per_episode=10,
    device="cpu",
    tensorboard_dir=None,
    batch_size=16,
    memory_capacity=1000,
    gamma=0.99,
    lr=0.001,
    agent_config={'epsilon_start': 1.0, 'epsilon_end': 0.1, 'epsilon_decay': 0.95},
    gamma_coverage=30.0,
    step_penalty=-0.05,
)

# Force empty map
env.obstacle_grid = np.zeros((5, 5), dtype=np.float32)
env.obstacle_grid[0, :] = 1.0
env.obstacle_grid[-1, :] = 1.0
env.obstacle_grid[:, 0] = 1.0
env.obstacle_grid[:, -1] = 1.0

def empty_map_gen():
    return env.obstacle_grid.copy()

env.training_map_generators['empty'] = empty_map_gen
env.training_map_order = ['empty']

print("\nTraining 5 episodes...")
metrics, _, _, _, _ = train(env, episodes_per_type=5, verbose=True, save_interval=100, model_dir="./test_models")

print(f"\n{'='*60}")
print("METRICS CHECK:")
print(f"{'='*60}")
print(f"episode_coverage length: {len(metrics.episode_coverage)}")
print(f"episode_coverage values: {metrics.episode_coverage}")
print(f"episode_steps length: {len(metrics.episode_steps)}")
print(f"episode_rewards length: {len(metrics.episode_rewards)}")
print(f"episode_avg_loss length: {len(metrics.episode_avg_loss)}")
print(f"\nExpected: 5 entries in each list")
print(f"Actual: {len(metrics.episode_coverage)} entries")

if len(metrics.episode_coverage) == 5:
    print("\n✅ SUCCESS: Episode metrics recording works!")
else:
    print(f"\n❌ FAILURE: Expected 5 episodes, got {len(metrics.episode_coverage)}")
