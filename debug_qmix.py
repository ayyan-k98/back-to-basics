"""
QMIX Debug Script - Diagnose Learning Failures
===============================================

This script runs extensive diagnostics to identify why QMIX is not learning.

Issues to investigate:
1. Why only 15/50 episodes completed
2. Why coverage isn't increasing (6.35 → 8.32)
3. Why loss is flat (0.0918 → 0.0918)
4. Whether replay buffer is filling
5. Whether optimization is running
6. Whether Q-values are updating
"""

import numpy as np
import torch
from environment import MARL_QMIX_Environment
from config import Config
import sys
import traceback


def debug_episode_termination():
    """Debug #1: Why are episodes terminating early?"""
    print("\n" + "="*80)
    print("DEBUG #1: EPISODE TERMINATION ANALYSIS")
    print("="*80)
    
    env = MARL_QMIX_Environment(
        grid_size=5,
        num_agents=2,
        sensor_range=3,
        comm_range=5.0,
        coverage_threshold=0.8,
        completion_threshold_perc=80.0,  # Lower threshold
        max_episodes=5,  # Just 5 episodes for debug
        max_steps_per_episode=15,
        device="cpu",
        tensorboard_dir=None,  # Disable tensorboard for debug
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
    
    # Force empty map (only borders are obstacles)
    env.obstacle_grid = np.zeros((5, 5), dtype=np.float32)
    env.obstacle_grid[0, :] = 1.0
    env.obstacle_grid[-1, :] = 1.0
    env.obstacle_grid[:, 0] = 1.0
    env.obstacle_grid[:, -1] = 1.0
    
    print(f"\nEnvironment setup:")
    print(f"  Grid size: {env.grid_size}x{env.grid_size}")
    print(f"  Free cells: {np.sum(env.obstacle_grid == 0)}")
    print(f"  Num agents: {env.num_agents}")
    print(f"  Max steps per episode: {env.max_steps_per_episode}")
    print(f"  Completion threshold: {env.completion_threshold_perc}%")
    
    episode_lengths = []
    episode_termination_reasons = []
    
    for episode in range(5):
        print(f"\n{'─'*80}")
        print(f"EPISODE {episode}")
        print(f"{'─'*80}")
        
        states = env.reset(full_reset=(episode == 0), mode='train')
        print(f"Initial positions: {env.agent_positions}")
        print(f"Initial coverage: {env.calculate_coverage_percentage():.1f}%")
        
        episode_done = False
        step_count = 0
        
        for step in range(env.max_steps_per_episode):
            step_count = step + 1
            
            # Get actions
            actions = {}
            for agent_id, agent in enumerate(env.agents):
                valid_actions = env.get_valid_actions(agent_id)
                action = agent.select_action(states[agent_id], valid_actions)
                actions[agent_id] = action
            
            # Step environment
            results = env.step(actions)
            
            # Extract done flags
            done_flags = {agent_id: result[3] for agent_id, result in results.items()}
            episode_done = any(done_flags.values())
            
            coverage = env.calculate_coverage_percentage()
            
            print(f"  Step {step:2d}: actions={actions}, coverage={coverage:5.1f}%, done={done_flags}")
            
            if episode_done:
                print(f"  → Episode TERMINATED at step {step}")
                print(f"  → Done flags: {done_flags}")
                print(f"  → Final coverage: {coverage:.1f}%")
                
                # Determine termination reason
                if coverage >= env.completion_threshold_perc:
                    reason = "COVERAGE_COMPLETE"
                elif step >= env.max_steps_per_episode - 1:
                    reason = "MAX_STEPS"
                else:
                    reason = "UNKNOWN"
                
                episode_termination_reasons.append(reason)
                break
            
            # Update states
            for agent_id, result in results.items():
                states[agent_id] = (result[0], result[1])
        
        if not episode_done:
            episode_termination_reasons.append("MAX_STEPS")
            print(f"  → Episode completed MAX_STEPS ({env.max_steps_per_episode})")
        
        episode_lengths.append(step_count)
        print(f"Episode {episode} length: {step_count} steps")
    
    print(f"\n{'='*80}")
    print("EPISODE TERMINATION SUMMARY")
    print(f"{'='*80}")
    print(f"Episode lengths: {episode_lengths}")
    print(f"Avg length: {np.mean(episode_lengths):.1f} steps")
    print(f"Termination reasons: {episode_termination_reasons}")
    print(f"  COVERAGE_COMPLETE: {episode_termination_reasons.count('COVERAGE_COMPLETE')}")
    print(f"  MAX_STEPS: {episode_termination_reasons.count('MAX_STEPS')}")
    print(f"  UNKNOWN: {episode_termination_reasons.count('UNKNOWN')}")
    
    return env


def debug_replay_buffer(env):
    """Debug #2: Is the replay buffer filling?"""
    print("\n" + "="*80)
    print("DEBUG #2: REPLAY BUFFER ANALYSIS")
    print("="*80)
    
    print(f"\nReplay buffer status:")
    print(f"  Current size: {len(env.memory)}")
    print(f"  Capacity: {env.memory.capacity}")
    print(f"  Batch size: {env.batch_size}")
    print(f"  Can optimize: {len(env.memory) >= env.batch_size}")
    
    if len(env.memory) > 0:
        # Sample one transition to check structure
        sample = env.memory.memory[0]
        print(f"\nSample transition structure:")
        print(f"  States shape: {[s.shape if hasattr(s, 'shape') else type(s) for s in sample.states]}")
        print(f"  Actions: {sample.actions}")
        print(f"  Rewards: {sample.rewards}")
        print(f"  Next states shape: {[s.shape if hasattr(s, 'shape') else type(s) for s in sample.next_states]}")
        print(f"  Dones: {sample.dones}")
    else:
        print("  WARNING: Buffer is EMPTY!")


def debug_optimization(env):
    """Debug #3: Is optimization running and updating weights?"""
    print("\n" + "="*80)
    print("DEBUG #3: OPTIMIZATION ANALYSIS")
    print("="*80)
    
    print(f"\nOptimization status:")
    print(f"  Global optimization steps: {env.global_optimization_steps}")
    print(f"  Buffer size: {len(env.memory)}")
    print(f"  Batch size: {env.batch_size}")
    
    if len(env.memory) < env.batch_size:
        print(f"  WARNING: Cannot optimize yet (need {env.batch_size - len(env.memory)} more transitions)")
        return None
    
    # Get initial weights
    initial_weights = {}
    for agent_id, agent in enumerate(env.agents):
        initial_weights[agent_id] = agent.policy_net.state_dict()['fc3.weight'].clone()
    initial_mixer_weights = env.mixer.hyper_w_1.state_dict()['weight'].clone()
    
    print(f"\nRunning 10 optimization steps...")
    losses = []
    
    for i in range(10):
        loss = env.optimize_qmix()
        if loss is not None:
            losses.append(loss)
            print(f"  Step {i+1}: loss = {loss:.6f}")
        else:
            print(f"  Step {i+1}: optimization SKIPPED")
    
    # Check if weights changed
    print(f"\nWeight update analysis:")
    for agent_id, agent in enumerate(env.agents):
        new_weights = agent.policy_net.state_dict()['fc3.weight']
        weight_diff = torch.abs(new_weights - initial_weights[agent_id]).max().item()
        print(f"  Agent {agent_id} max weight change: {weight_diff:.6e}")
    
    new_mixer_weights = env.mixer.hyper_w_1.state_dict()['weight']
    mixer_diff = torch.abs(new_mixer_weights - initial_mixer_weights).max().item()
    print(f"  Mixer max weight change: {mixer_diff:.6e}")
    
    if losses:
        print(f"\nLoss statistics:")
        print(f"  First loss: {losses[0]:.6f}")
        print(f"  Last loss: {losses[-1]:.6f}")
        print(f"  Mean loss: {np.mean(losses):.6f}")
        print(f"  Loss change: {losses[-1] - losses[0]:.6f}")
    
    return losses


def debug_q_values(env):
    """Debug #4: Are Q-values changing over time?"""
    print("\n" + "="*80)
    print("DEBUG #4: Q-VALUE ANALYSIS")
    print("="*80)
    
    # Get a state
    states = env.reset(full_reset=False, mode='train')
    
    print(f"\nQ-value analysis for each agent:")
    for agent_id, agent in enumerate(env.agents):
        grid, features = states[agent_id]
        
        # Get Q-values
        with torch.no_grad():
            grid_tensor = torch.FloatTensor(grid).unsqueeze(0).to(agent.device)
            feature_tensor = torch.FloatTensor(features).unsqueeze(0).to(agent.device)
            q_values = agent.policy_net(grid_tensor, feature_tensor).cpu().numpy()[0]
        
        print(f"\n  Agent {agent_id}:")
        print(f"    Q-values: {q_values}")
        print(f"    Min: {q_values.min():.4f}")
        print(f"    Max: {q_values.max():.4f}")
        print(f"    Mean: {q_values.mean():.4f}")
        print(f"    Std: {q_values.std():.4f}")
        print(f"    Best action: {q_values.argmax()}")


def debug_coverage_calculation():
    """Debug #5: Is coverage calculation correct?"""
    print("\n" + "="*80)
    print("DEBUG #5: COVERAGE CALCULATION ANALYSIS")
    print("="*80)
    
    env = MARL_QMIX_Environment(
        grid_size=5,
        num_agents=2,
        sensor_range=3,
        comm_range=5.0,
        coverage_threshold=0.8,
        completion_threshold_perc=80.0,
        max_episodes=1,
        max_steps_per_episode=15,
        device="cpu",
        tensorboard_dir=None,
        batch_size=32,
        memory_capacity=5000,
        gamma=0.99,
        lr=0.001,
        agent_config={
            'epsilon_start': 0.0,  # No exploration for deterministic test
            'epsilon_end': 0.0,
            'epsilon_decay': 1.0,
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
    
    free_cells = np.sum(env.obstacle_grid == 0)
    print(f"\nGrid analysis:")
    print(f"  Total cells: {env.grid_size * env.grid_size}")
    print(f"  Free cells: {free_cells}")
    print(f"  Obstacle cells: {np.sum(env.obstacle_grid == 1)}")
    
    # Reset and check initial coverage
    states = env.reset(full_reset=True, mode='train')
    
    print(f"\nInitial state:")
    print(f"  Agent positions: {env.agent_positions}")
    print(f"  Coverage percentage: {env.calculate_coverage_percentage():.2f}%")
    
    # Check coverage grid
    print(f"\nCoverage grid (1 = covered, 0 = not covered):")
    print(env.coverage_grid.astype(int))
    
    covered_cells = np.sum(env.coverage_grid == 1.0)
    print(f"\n  Covered cells: {covered_cells}")
    print(f"  Coverage %: {100.0 * covered_cells / free_cells:.2f}%")
    
    # Manually move agents to cover specific areas
    print(f"\nMoving agents to test coverage updates...")
    env.agent_positions[0] = np.array([2, 2])
    env.agent_positions[1] = np.array([3, 3])
    env.update_coverage()
    
    print(f"\nAfter manual positioning:")
    print(f"  Agent positions: {env.agent_positions}")
    print(f"  Coverage percentage: {env.calculate_coverage_percentage():.2f}%")
    print(f"\nCoverage grid:")
    print(env.coverage_grid.astype(int))


def debug_training_loop():
    """Debug #6: Full training loop with detailed logging"""
    print("\n" + "="*80)
    print("DEBUG #6: FULL TRAINING LOOP ANALYSIS")
    print("="*80)
    
    env = MARL_QMIX_Environment(
        grid_size=5,
        num_agents=2,
        sensor_range=3,
        comm_range=5.0,
        coverage_threshold=0.8,
        completion_threshold_perc=80.0,
        max_episodes=10,  # 10 episodes for debug
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
    
    print(f"\nStarting training loop...")
    print(f"  Episodes: {env.max_episodes}")
    print(f"  Steps per episode: {env.max_steps_per_episode}")
    
    episode_coverages = []
    episode_rewards = []
    optimization_counts = []
    
    for episode in range(env.max_episodes):
        states = env.reset(full_reset=(episode == 0), mode='train')
        episode_reward = 0
        optimizations_this_episode = 0
        
        for step in range(env.max_steps_per_episode):
            # Select actions
            actions = {}
            for agent_id, agent in enumerate(env.agents):
                valid_actions = env.get_valid_actions(agent_id)
                action = agent.select_action(states[agent_id], valid_actions)
                actions[agent_id] = action
            
            # Step
            results = env.step(actions)
            
            # Accumulate rewards
            for agent_id, result in results.items():
                episode_reward += result[2]
            
            # Check if optimization happened
            buffer_size_before = env.global_optimization_steps
            loss = env.optimize_qmix()
            if loss is not None:
                optimizations_this_episode += 1
            
            # Check done
            if any(result[3] for result in results.values()):
                break
            
            # Update states
            for agent_id, result in results.items():
                states[agent_id] = (result[0], result[1])
        
        final_coverage = env.calculate_coverage_percentage()
        episode_coverages.append(final_coverage)
        episode_rewards.append(episode_reward)
        optimization_counts.append(optimizations_this_episode)
        
        epsilon = env.agents[0].epsilon
        print(f"Episode {episode:2d}: coverage={final_coverage:5.1f}%, "
              f"reward={episode_reward:7.2f}, "
              f"optimizations={optimizations_this_episode:3d}, "
              f"buffer={len(env.memory):4d}, "
              f"epsilon={epsilon:.3f}")
    
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Coverage progression:")
    print(f"  First 3 episodes: {episode_coverages[:3]}")
    print(f"  Last 3 episodes: {episode_coverages[-3:]}")
    print(f"  Mean (first 3): {np.mean(episode_coverages[:3]):.2f}%")
    print(f"  Mean (last 3): {np.mean(episode_coverages[-3:]):.2f}%")
    print(f"  Improvement: {np.mean(episode_coverages[-3:]) - np.mean(episode_coverages[:3]):.2f}%")
    
    print(f"\nReward progression:")
    print(f"  Mean (first 3): {np.mean(episode_rewards[:3]):.2f}")
    print(f"  Mean (last 3): {np.mean(episode_rewards[-3:]):.2f}")
    print(f"  Improvement: {np.mean(episode_rewards[-3:]) - np.mean(episode_rewards[:3]):.2f}")
    
    print(f"\nOptimization counts: {optimization_counts}")
    print(f"  Total optimizations: {sum(optimization_counts)}")
    print(f"  Final buffer size: {len(env.memory)}")
    print(f"  Global optimization steps: {env.global_optimization_steps}")


def main():
    """Run all debug tests"""
    print("\n" + "#"*80)
    print("# QMIX COMPREHENSIVE DEBUGGING SUITE")
    print("#"*80)
    print(f"# Date: November 13, 2025")
    print(f"# Purpose: Diagnose why QMIX is not learning")
    print("#"*80)
    
    try:
        # Debug 1: Episode termination
        env = debug_episode_termination()
        
        # Debug 2: Replay buffer
        debug_replay_buffer(env)
        
        # Debug 3: Optimization
        debug_optimization(env)
        
        # Debug 4: Q-values
        debug_q_values(env)
        
        # Debug 5: Coverage calculation
        debug_coverage_calculation()
        
        # Debug 6: Full training loop
        debug_training_loop()
        
        print("\n" + "#"*80)
        print("# DEBUG SUITE COMPLETED SUCCESSFULLY")
        print("#"*80)
        
    except Exception as e:
        print("\n" + "!"*80)
        print("! EXCEPTION OCCURRED DURING DEBUG")
        print("!"*80)
        print(f"\nError: {e}")
        print(f"\nTraceback:")
        traceback.print_exc()
        print("\n" + "!"*80)


if __name__ == "__main__":
    main()
