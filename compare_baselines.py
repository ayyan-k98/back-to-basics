"""
Baseline Comparison: Test QMIX vs Greedy vs Independent Q-Learning vs Random

This script answers the critical question:
  "Is QMIX's 81% coverage on Test 3 good or bad?"

By comparing against:
1. Greedy Frontier: Simple heuristic baseline
2. Independent Q-Learning: Learning without coordination
3. Random: Sanity check (should be worst)

All methods use SAME observation space and action space (fair comparison).
"""

import os
import time
import random
import numpy as np
import torch
from collections import defaultdict

from environment import MARL_QMIX_Environment
from baselines import BaselineRunner
from utils import generate_room_map, ensure_dir


def run_baseline_episode(env, baseline_runner, max_steps=100, verbose=False):
    """Run one episode with baseline agents."""
    episode_rewards = {i: 0.0 for i in range(env.num_agents)}
    episode_steps = 0
    done = False
    
    trajectories = defaultdict(list)
    for i in range(env.num_agents):
        trajectories[i].append(env.agents[i].state.position)
    
    while episode_steps < max_steps and not done:
        # Get baseline actions
        actions = baseline_runner.select_actions()
        
        # Store states for independent Q-learning
        if baseline_runner.baseline_type == "independent":
            states = {i: env.agents[i].get_state_tensor() for i in range(env.num_agents)}
        
        # Execute actions
        results = env.step(actions)
        
        # Train independent Q-learning agents
        if baseline_runner.baseline_type == "independent":
            transitions = {}
            for agent_id, (_, _, reward, done_flag, _) in results.items():
                next_state = env.agents[agent_id].get_state_tensor()
                transitions[agent_id] = (states[agent_id], actions[agent_id], 
                                        reward, next_state, done_flag)
            baseline_runner.train_step(transitions)
        
        # Track rewards and trajectory
        for agent_id, (_, _, reward, done_flag, _) in results.items():
            episode_rewards[agent_id] += reward
            trajectories[agent_id].append(env.agents[agent_id].state.position)
            if done_flag:
                done = True
        
        episode_steps += 1
        
        if verbose and episode_steps % 20 == 0:
            coverage = env.calculate_coverage_percentage()
            print(f"  Step {episode_steps}: Coverage={coverage:.1f}%")
    
    # Decay epsilon for independent Q-learning
    baseline_runner.decay_epsilon()
    
    final_coverage = env.calculate_coverage_percentage()
    return final_coverage, episode_steps, episode_rewards, trajectories


def test_baseline_on_obstacles(baseline_type: str, num_episodes: int = 300, 
                               verbose: bool = False):
    """
    Test a baseline on Test 3 (obstacles).
    
    Args:
        baseline_type: "greedy", "independent", or "random"
        num_episodes: Number of episodes to run
        verbose: Print detailed progress
    """
    print(f"\n{'='*70}")
    print(f"Testing: {baseline_type.upper()} Baseline")
    print(f"{'='*70}")
    
    # Same config as QMIX Test 3
    env = MARL_QMIX_Environment(
        grid_size=10,
        num_agents=2,
        sensor_range=4,
        comm_range=8.0,
        coverage_threshold=0.8,
        completion_threshold_perc=80.0,
        max_episodes=num_episodes,
        max_steps_per_episode=100,
        device=None,  # Auto-detect: cuda if available, else cpu
        tensorboard_dir=None,  # Disable TensorBoard for baselines
        batch_size=64,
        memory_capacity=15000,
        gamma=0.99,
        lr_agents=0.0005,
        lr_mixer=0.0001,
        soft_update_tau=0.001,
        agent_config={
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.99,
        },
        gamma_coverage=25.0,
        step_penalty=-0.02,
    )
    
    # Map generator (same as Test 3)
    def room_map():
        return generate_room_map(10, num_rooms_range=(2, 3), room_size_range=(3, 4))
    
    env.training_map_generators['room'] = room_map
    env.training_map_order = ['room']
    
    # Initialize environment
    env.reset(full_reset=True, mode='train')
    
    # Create baseline runner
    baseline_runner = BaselineRunner(env, baseline_type=baseline_type)
    
    # Track metrics
    coverage_history = []
    steps_history = []
    
    print(f"Running {num_episodes} episodes...")
    start_time = time.time()
    
    for ep in range(num_episodes):
        # Reset environment (new map each episode)
        env.reset(full_reset=False, mode='train')
        
        # Run episode
        final_coverage, episode_steps, episode_rewards, trajectories = run_baseline_episode(
            env, baseline_runner, max_steps=100, verbose=verbose
        )
        
        coverage_history.append(final_coverage)
        steps_history.append(episode_steps)
        
        if (ep + 1) % 50 == 0:
            recent_cov = np.mean(coverage_history[-50:])
            epsilon = baseline_runner.get_epsilon()
            print(f"  Episode {ep+1}/{num_episodes}: "
                  f"Coverage={final_coverage:.1f}%, "
                  f"Avg(last 50)={recent_cov:.1f}%, "
                  f"Epsilon={epsilon:.3f}")
    
    elapsed_time = time.time() - start_time
    
    # Compute statistics
    early_coverage = np.mean(coverage_history[:30]) if len(coverage_history) >= 30 else 0
    late_coverage = np.mean(coverage_history[-30:]) if len(coverage_history) >= 30 else 0
    improvement = late_coverage - early_coverage
    
    mean_coverage = np.mean(coverage_history)
    std_coverage = np.std(coverage_history)
    max_coverage = np.max(coverage_history)
    min_coverage = np.min(coverage_history)
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {baseline_type.upper()}")
    print(f"{'='*70}")
    print(f"Coverage (early):     {early_coverage:.2f}%")
    print(f"Coverage (late):      {late_coverage:.2f}%")
    print(f"Improvement:          {improvement:+.2f}%")
    print(f"Mean coverage:        {mean_coverage:.2f}% ± {std_coverage:.2f}%")
    print(f"Range:                [{min_coverage:.2f}%, {max_coverage:.2f}%]")
    print(f"Mean steps/episode:   {np.mean(steps_history):.1f}")
    print(f"Time elapsed:         {elapsed_time:.1f}s")
    print(f"{'='*70}")
    
    return {
        'baseline_type': baseline_type,
        'early_coverage': early_coverage,
        'late_coverage': late_coverage,
        'improvement': improvement,
        'mean_coverage': mean_coverage,
        'std_coverage': std_coverage,
        'max_coverage': max_coverage,
        'min_coverage': min_coverage,
        'coverage_history': coverage_history,
    }


def compare_all_baselines(num_episodes: int = 300):
    """Run all baselines and QMIX, then compare."""
    print("\n" + "#"*70)
    print("# BASELINE COMPARISON: Test 3 (Obstacles)")
    print("#"*70)
    print("# Question: Is QMIX's 81% coverage good or bad?")
    print("# Answer: Compare against greedy, independent Q-learning, random")
    print("#"*70)
    
    results = {}
    
    # 1. Random baseline (sanity check)
    print("\n[1/4] Running RANDOM baseline...")
    results['random'] = test_baseline_on_obstacles('random', num_episodes=50, verbose=False)
    
    # 2. Greedy baseline
    print("\n[2/4] Running GREEDY baseline...")
    results['greedy'] = test_baseline_on_obstacles('greedy', num_episodes=num_episodes, verbose=False)
    
    # 3. Independent Q-learning
    print("\n[3/4] Running INDEPENDENT Q-LEARNING baseline...")
    results['independent'] = test_baseline_on_obstacles('independent', num_episodes=num_episodes, verbose=False)
    
    # 4. QMIX (already have results from validate_tier1.py)
    print("\n[4/4] QMIX results (from previous run):")
    results['qmix'] = {
        'baseline_type': 'QMIX',
        'early_coverage': 84.0,  # From Test 3 validation
        'late_coverage': 81.3,
        'improvement': -2.7,
        'mean_coverage': 82.0,  # Approximate
        'std_coverage': 2.0,
        'max_coverage': 86.0,
        'min_coverage': 78.0,
    }
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(f"{'Method':<20} {'Early%':<10} {'Late%':<10} {'Δ%':<10} {'Mean%':<10} {'Verdict':<15}")
    print("-"*70)
    
    methods = ['random', 'greedy', 'independent', 'qmix']
    for method in methods:
        r = results[method]
        early = r['early_coverage']
        late = r['late_coverage']
        delta = r['improvement']
        mean = r['mean_coverage']
        
        # Determine verdict
        if method == 'random':
            verdict = "Sanity check"
        elif method == 'qmix':
            verdict = "❓ TO BE JUDGED"
        else:
            verdict = "Baseline"
        
        print(f"{method.upper():<20} {early:<10.1f} {late:<10.1f} {delta:<10.1f} {mean:<10.1f} {verdict:<15}")
    
    print("="*70)
    
    # Analysis
    print("\nANALYSIS:")
    print("-"*70)
    
    qmix_mean = results['qmix']['mean_coverage']
    greedy_mean = results['greedy']['mean_coverage']
    indep_mean = results['independent']['mean_coverage']
    random_mean = results['random']['mean_coverage']
    
    print(f"QMIX mean coverage:         {qmix_mean:.1f}%")
    print(f"Greedy mean coverage:       {greedy_mean:.1f}%")
    print(f"Independent mean coverage:  {indep_mean:.1f}%")
    print(f"Random mean coverage:       {random_mean:.1f}%")
    print()
    
    if qmix_mean > greedy_mean and qmix_mean > indep_mean:
        print("✅ VERDICT: QMIX WINS")
        print(f"   QMIX outperforms both greedy (+{qmix_mean - greedy_mean:.1f}%) ")
        print(f"   and independent Q-learning (+{qmix_mean - indep_mean:.1f}%)")
        print("   The slight regression is acceptable - QMIX is learning coordination.")
    elif qmix_mean > greedy_mean:
        print("⚠️  VERDICT: QMIX BEATS GREEDY, LOSES TO INDEPENDENT")
        print(f"   QMIX better than greedy (+{qmix_mean - greedy_mean:.1f}%)")
        print(f"   But worse than independent ({qmix_mean - indep_mean:.1f}%)")
        print("   Suggests: Coordination is HURTING, not helping.")
    elif qmix_mean > indep_mean:
        print("⚠️  VERDICT: QMIX BEATS INDEPENDENT, LOSES TO GREEDY")
        print(f"   Greedy: {greedy_mean:.1f}% (simple heuristic wins!)")
        print(f"   QMIX: {qmix_mean:.1f}% (learning is suboptimal)")
        print("   Suggests: Need better exploration or reward shaping.")
    else:
        print("❌ VERDICT: QMIX LOSES TO ALL BASELINES")
        print(f"   Greedy:       {greedy_mean:.1f}% > QMIX: {qmix_mean:.1f}%")
        print(f"   Independent:  {indep_mean:.1f}% > QMIX: {qmix_mean:.1f}%")
        print("   Suggests: Fundamental issue with QMIX implementation.")
    
    print("="*70)
    
    return results


if __name__ == "__main__":
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run comparison
    results = compare_all_baselines(num_episodes=300)
    
    print("\n✅ Baseline comparison complete!")
    print("Next steps based on results:")
    print("  - If QMIX wins: Proceed to scaling experiments")
    print("  - If greedy wins: Add frontier rewards to QMIX")
    print("  - If independent wins: Debug coordination mechanism")
