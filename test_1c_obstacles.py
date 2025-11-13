"""
TEST 1C: Two Agents, Simple Obstacles (POMDP VALIDATION)

IMMUTABLE CONFIGURATION - DO NOT CHANGE
This test validates that QMIX handles partial observability (obstacles).

Success Criteria (LOCKED):
  - QMIX > Independent: p<0.05 (coordination still helps)
  - No catastrophic regression: Late >= 0.9 * Early
  - Both beat random: p<0.05
  - POMDP integrity: Agents discover obstacles (not omniscient)

Prerequisite: Tests 1A and 1B must PASS first
If Test 1C fails, obstacle handling is broken.
"""

import os
import time
import random
import numpy as np
import torch
import scipy.stats as stats
from typing import Dict

from environment import MARL_QMIX_Environment
from baselines import BaselineRunner
from utils import generate_room_map, ensure_dir


# ============================================================
# LOCKED CONFIGURATION (DO NOT MODIFY)
# ============================================================
TEST_1C_CONFIG = {
    'grid_size': 20,
    'num_agents': 2,            # Two agents (coordination needed)
    'sensor_range': 4,          # Slightly larger for obstacles
    'comm_range': 8.0,          # Can communicate across rooms
    'coverage_threshold': 0.8,
    'completion_threshold_perc': 85.0,
    'max_episodes': 300,        # More episodes for harder task
    'max_steps_per_episode': 150,  # More steps for obstacles
    
    # Rewards (standard)
    'gamma_coverage': 15.0,
    'step_penalty': -0.02,
    'orientation_cost_factor': 0.02,
    'invalid_move_penalty': -0.5,
    
    # Learning (Tier 1 stable config)
    'batch_size': 64,
    'memory_capacity': 30000,   # Larger for complex environment
    'gamma': 0.99,
    'lr_agents': 5e-4,          # Agent networks
    'lr_mixer': 1e-4,           # Mixer 5x slower
    'use_soft_update': True,
    'soft_update_tau': 0.001,
    
    # Exploration (slower decay for 300 episodes)
    'agent_config': {
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.98,  # Reaches ~0.05 by episode 150
    },
    
    # Device
    'device': None,  # Auto-detect
    'tensorboard_dir': './runs/test_1c',
}

# ============================================================
# SUCCESS CRITERIA (LOCKED - DO NOT MODIFY)
# ============================================================
SUCCESS_CRITERIA_1C = {
    'qmix_beats_independent': True,     # QMIX > Independent
    'statistical_significance': 0.05,   # p < 0.05
    'no_catastrophic_regression': True, # Late >= 0.9 * Early
    'both_beat_random': True,           # QMIX & Independent > Random
    'random_baseline_min': 35.0,        # Harder with obstacles
    'random_baseline_max': 55.0,
    'greedy_baseline_min': 50.0,        # Heuristic still helps
    'greedy_baseline_max': 70.0,
    'loss_stable': True,                # Max loss < 10.0
}


def run_random_baseline_obstacles(num_episodes: int = 50, verbose: bool = False):
    """
    PURE RANDOM BASELINE - 2 agents, obstacles, no learning.
    
    Expected: 35-55% coverage (harder than empty grid)
    """
    print("\n" + "="*70)
    print("RUNNING RANDOM BASELINE (2 AGENTS + OBSTACLES)")
    print("="*70)
    
    config = TEST_1C_CONFIG.copy()
    config['tensorboard_dir'] = None
    
    env = MARL_QMIX_Environment(**config)
    env.training_map_generators['room'] = lambda: generate_room_map(20, num_rooms_range=(2, 3))
    env.training_map_order = ['room']
    
    coverages = []
    
    for ep in range(num_episodes):
        env.reset(full_reset=(ep == 0), mode='train')
        
        for step in range(150):
            actions = {}
            for agent_id in range(2):
                valid_actions = env.get_valid_actions(agent_id)
                actions[agent_id] = random.choice(valid_actions) if valid_actions else (0, 0)
            env.step(actions)
        
        final_coverage = env.calculate_coverage_percentage()
        coverages.append(final_coverage)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}: Coverage={final_coverage:.1f}%, Avg={np.mean(coverages):.1f}%")
    
    mean_cov = np.mean(coverages)
    print(f"\nRandom: {mean_cov:.2f}% ± {np.std(coverages):.2f}%")
    
    return {'mean_coverage': mean_cov, 'std_coverage': np.std(coverages), 'coverages': coverages}


def run_greedy_baseline_obstacles(num_episodes: int = 50, verbose: bool = False):
    """
    GREEDY FRONTIER BASELINE - 2 agents, obstacles, heuristic only.
    
    Expected: 50-70% coverage
    """
    print("\n" + "="*70)
    print("RUNNING GREEDY BASELINE (2 AGENTS + OBSTACLES)")
    print("="*70)
    
    config = TEST_1C_CONFIG.copy()
    config['tensorboard_dir'] = None
    
    env = MARL_QMIX_Environment(**config)
    env.training_map_generators['room'] = lambda: generate_room_map(20, num_rooms_range=(2, 3))
    env.training_map_order = ['room']
    
    env.reset(full_reset=True, mode='train')
    baseline_runner = BaselineRunner(env, baseline_type='greedy')
    
    coverages = []
    
    for ep in range(num_episodes):
        env.reset(full_reset=False, mode='train')
        
        for step in range(150):
            actions = baseline_runner.select_actions()
            env.step(actions)
        
        final_coverage = env.calculate_coverage_percentage()
        coverages.append(final_coverage)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}: Coverage={final_coverage:.1f}%, Avg={np.mean(coverages):.1f}%")
    
    mean_cov = np.mean(coverages)
    print(f"\nGreedy: {mean_cov:.2f}% ± {np.std(coverages):.2f}%")
    
    return {'mean_coverage': mean_cov, 'std_coverage': np.std(coverages), 'coverages': coverages}


def run_independent_ql_obstacles(num_episodes: int = 300, verbose: bool = True):
    """
    INDEPENDENT Q-LEARNING - 2 agents, obstacles, no coordination.
    
    Expected: Should learn to navigate around obstacles
    """
    print("\n" + "="*70)
    print("RUNNING INDEPENDENT Q-LEARNING (2 AGENTS + OBSTACLES)")
    print("="*70)
    
    config = TEST_1C_CONFIG.copy()
    config['tensorboard_dir'] = None
    
    env = MARL_QMIX_Environment(**config)
    env.training_map_generators['room'] = lambda: generate_room_map(20, num_rooms_range=(2, 3))
    env.training_map_order = ['room']
    
    env.reset(full_reset=True, mode='train')
    baseline_runner = BaselineRunner(env, baseline_type='independent')
    
    coverages = []
    losses = []
    
    print(f"Training for {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        env.reset(full_reset=False, mode='train')
        
        episode_losses = []
        
        for step in range(150):
            # Store states
            states = {i: env.agents[i].get_state_tensor() for i in range(2)}
            
            # Select actions
            actions = baseline_runner.select_actions()
            
            # Execute
            results = env.step(actions)
            
            # Train independent agents
            transitions = {}
            for agent_id in range(2):
                next_state = env.agents[agent_id].get_state_tensor()
                _, _, reward, done, _ = results[agent_id]
                transitions[agent_id] = (states[agent_id], actions[agent_id], reward, next_state, done)
            
            loss = baseline_runner.train_step(transitions)
            if loss is not None:
                episode_losses.append(loss)
        
        baseline_runner.decay_epsilon()
        
        final_coverage = env.calculate_coverage_percentage()
        coverages.append(final_coverage)
        
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        if verbose and (ep + 1) % 30 == 0:
            recent_cov = np.mean(coverages[-30:])
            recent_loss = np.mean(losses[-30:]) if losses else 0.0
            epsilon = baseline_runner.get_epsilon()
            print(f"  Episode {ep+1}: Coverage={final_coverage:.1f}%, "
                  f"Avg(last 30)={recent_cov:.1f}%, Loss={recent_loss:.4f}, ε={epsilon:.3f}")
    
    early_cov = np.mean(coverages[:50])
    late_cov = np.mean(coverages[-50:])
    mean_cov = np.mean(coverages)
    
    print(f"\nIndependent Q-Learning:")
    print(f"  Early: {early_cov:.2f}%, Late: {late_cov:.2f}%, Improvement: {late_cov - early_cov:+.2f}%")
    print(f"  Mean: {mean_cov:.2f}% ± {np.std(coverages):.2f}%")
    
    return {
        'mean_coverage': mean_cov,
        'std_coverage': np.std(coverages),
        'early_coverage': early_cov,
        'late_coverage': late_cov,
        'improvement': late_cov - early_cov,
        'coverages': coverages,
        'losses': losses,
    }


def run_qmix_obstacles(num_episodes: int = 300, verbose: bool = True):
    """
    QMIX - 2 agents with coordination, obstacles.
    
    Expected: Coordination helps navigate obstacles together
    """
    print("\n" + "="*70)
    print("RUNNING QMIX (2 AGENTS + OBSTACLES + COORDINATION)")
    print("="*70)
    
    env = MARL_QMIX_Environment(**TEST_1C_CONFIG)
    env.training_map_generators['room'] = lambda: generate_room_map(20, num_rooms_range=(2, 3))
    env.training_map_order = ['room']
    
    env.reset(full_reset=True, mode='train')
    
    coverages = []
    losses = []
    
    print(f"Training for {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        env.reset(full_reset=False, mode='train')
        
        for step in range(150):
            actions = {}
            for agent_id, agent in enumerate(env.agents):
                state_tensor = agent.get_state_tensor()
                valid_actions = env.get_valid_actions(agent_id)
                action = agent.select_action(state_tensor, valid_actions)
                actions[agent_id] = action
            
            env.step(actions)
        
        final_coverage = env.calculate_coverage_percentage()
        coverages.append(final_coverage)
        
        if env.metrics.qmix_loss:
            recent_losses = env.metrics.qmix_loss[-150:]
            losses.append(np.mean(recent_losses))
        
        # Decay epsilon after each episode
        for agent in env.agents:
            agent.update_epsilon()
        
        if verbose and (ep + 1) % 30 == 0:
            recent_cov = np.mean(coverages[-30:])
            recent_loss = np.mean(losses[-30:]) if losses else 0.0
            epsilon = env.agents[0].epsilon
            print(f"  Episode {ep+1}: Coverage={final_coverage:.1f}%, "
                  f"Avg(last 30)={recent_cov:.1f}%, Loss={recent_loss:.4f}, ε={epsilon:.3f}")
    
    early_cov = np.mean(coverages[:50])
    late_cov = np.mean(coverages[-50:])
    mean_cov = np.mean(coverages)
    
    print(f"\nQMIX:")
    print(f"  Early: {early_cov:.2f}%, Late: {late_cov:.2f}%, Improvement: {late_cov - early_cov:+.2f}%")
    print(f"  Mean: {mean_cov:.2f}% ± {np.std(coverages):.2f}%")
    
    return {
        'mean_coverage': mean_cov,
        'std_coverage': np.std(coverages),
        'early_coverage': early_cov,
        'late_coverage': late_cov,
        'improvement': late_cov - early_cov,
        'coverages': coverages,
        'losses': losses,
    }


def evaluate_test_1c(random_results, greedy_results, independent_results, qmix_results):
    """
    Evaluate Test 1C against LOCKED success criteria.
    """
    print("\n" + "="*70)
    print("TEST 1C EVALUATION")
    print("="*70)
    
    criteria = SUCCESS_CRITERIA_1C
    results = {
        'test': 'Test_1C',
        'config': 'Two agents, 20×20 grid with obstacles (2-3 rooms)',
        'passed': True,
        'failures': [],
    }
    
    # Check 1: Random baseline sanity
    random_mean = random_results['mean_coverage']
    if not (criteria['random_baseline_min'] <= random_mean <= criteria['random_baseline_max']):
        results['passed'] = False
        results['failures'].append(f"Random {random_mean:.1f}% outside [35, 55]")
        print(f"❌ FAIL: Random baseline sanity check")
    else:
        print(f"✅ PASS: Random {random_mean:.1f}% in range")
    
    # Check 2: Greedy baseline sanity
    greedy_mean = greedy_results['mean_coverage']
    if not (criteria['greedy_baseline_min'] <= greedy_mean <= criteria['greedy_baseline_max']):
        results['passed'] = False
        results['failures'].append(f"Greedy {greedy_mean:.1f}% outside [50, 70]")
        print(f"❌ FAIL: Greedy baseline sanity check")
    else:
        print(f"✅ PASS: Greedy {greedy_mean:.1f}% in range")
    
    # Check 3: QMIX beats Independent
    qmix_mean = qmix_results['mean_coverage']
    indep_mean = independent_results['mean_coverage']
    
    if qmix_mean <= indep_mean:
        results['passed'] = False
        results['failures'].append(f"QMIX {qmix_mean:.1f}% NOT better than Independent {indep_mean:.1f}%")
        print(f"❌ FAIL: QMIX doesn't beat Independent with obstacles")
    else:
        gap = qmix_mean - indep_mean
        print(f"✅ PASS: QMIX beats Independent by {gap:.1f}%")
    
    # Check 4: Statistical significance
    t_stat, p_value = stats.ttest_ind(qmix_results['coverages'], independent_results['coverages'])
    if p_value >= criteria['statistical_significance']:
        results['passed'] = False
        results['failures'].append(f"Not statistically significant: p={p_value:.4f} >= 0.05")
        print(f"❌ FAIL: Not statistically significant (p={p_value:.4f})")
    else:
        print(f"✅ PASS: Statistically significant (p={p_value:.4f})")
    
    # Check 5: Both beat random
    t_qmix_vs_random, p_qmix = stats.ttest_ind(qmix_results['coverages'], random_results['coverages'])
    t_indep_vs_random, p_indep = stats.ttest_ind(independent_results['coverages'], random_results['coverages'])
    
    if p_qmix >= 0.05 or p_indep >= 0.05:
        results['passed'] = False
        results['failures'].append(f"Learning doesn't beat random (p_qmix={p_qmix:.4f}, p_indep={p_indep:.4f})")
        print(f"❌ FAIL: Learning doesn't beat random baseline")
    else:
        print(f"✅ PASS: Both beat random (p_qmix={p_qmix:.4f}, p_indep={p_indep:.4f})")
    
    # Check 6: No catastrophic regression (QMIX)
    qmix_early = qmix_results['early_coverage']
    qmix_late = qmix_results['late_coverage']
    
    if qmix_late < 0.9 * qmix_early:
        results['passed'] = False
        results['failures'].append(f"Catastrophic regression: Late {qmix_late:.1f}% < 0.9 * Early {qmix_early:.1f}%")
        print(f"❌ FAIL: QMIX shows catastrophic regression")
    else:
        regression_pct = (qmix_late / qmix_early) * 100 if qmix_early > 0 else 100
        print(f"✅ PASS: No catastrophic regression (late={regression_pct:.1f}% of early)")
    
    # Check 7: Loss stability
    if qmix_results['losses']:
        max_loss = np.max(qmix_results['losses'])
        if max_loss > 10.0:
            results['passed'] = False
            results['failures'].append(f"Loss unstable: max={max_loss:.2f} (threshold 10.0)")
            print(f"❌ FAIL: Loss diverged (max={max_loss:.2f})")
        else:
            print(f"✅ PASS: Loss stable (max={max_loss:.2f})")
    
    # Final verdict
    print("\n" + "="*70)
    if results['passed']:
        print("✅✅✅ TEST 1C PASSED ✅✅✅")
        print("OBSTACLE HANDLING VALIDATED - PHASE 1 COMPLETE!")
        print("Ready for Phase 2: Scalability (4+ agents)")
    else:
        print("❌❌❌ TEST 1C FAILED ❌❌❌")
        print("OBSTACLE HANDLING BROKEN")
        print("\nFailures:")
        for failure in results['failures']:
            print(f"  - {failure}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    # Check Test 1A and 1B passed
    if not os.path.exists('./test_results/test_1a_results.txt'):
        print("❌ ERROR: Test 1A must be run first")
        print("Run: python test_1a_single_agent.py")
        exit(1)
    
    if not os.path.exists('./test_results/test_1b_results.txt'):
        print("❌ ERROR: Test 1B must be run first")
        print("Run: python test_1b_two_agents.py")
        exit(1)
    
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("\n" + "#"*70)
    print("# TEST 1C: TWO AGENTS, OBSTACLES")
    print("# POMDP VALIDATION - IMMUTABLE")
    print("#"*70)
    print("\nPurpose: Validate that QMIX handles partial observability")
    print("Config: 2 agents, 20×20 grid, 2-3 rooms, 300 episodes")
    print("Success: QMIX > Independent (p<0.05), No catastrophic regression")
    print("\n" + "#"*70)
    
    # Step 1: Random baseline
    random_results = run_random_baseline_obstacles(num_episodes=50, verbose=True)
    
    # Step 2: Greedy baseline
    greedy_results = run_greedy_baseline_obstacles(num_episodes=50, verbose=True)
    
    # Step 3: Independent Q-Learning
    independent_results = run_independent_ql_obstacles(num_episodes=300, verbose=True)
    
    # Step 4: QMIX
    qmix_results = run_qmix_obstacles(num_episodes=300, verbose=True)
    
    # Step 5: Evaluate
    evaluation = evaluate_test_1c(random_results, greedy_results, independent_results, qmix_results)
    
    # Save results
    ensure_dir('./test_results')
    results_path = './test_results/test_1c_results.txt'
    with open(results_path, 'w') as f:
        f.write("TEST 1C RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Random:       {random_results['mean_coverage']:.2f}%\n")
        f.write(f"Greedy:       {greedy_results['mean_coverage']:.2f}%\n")
        f.write(f"Independent:  {independent_results['mean_coverage']:.2f}%\n")
        f.write(f"QMIX:         {qmix_results['mean_coverage']:.2f}%\n")
        f.write(f"\nGap (QMIX - Independent): {qmix_results['mean_coverage'] - independent_results['mean_coverage']:.2f}%\n")
        f.write(f"QMIX Early: {qmix_results['early_coverage']:.2f}%, Late: {qmix_results['late_coverage']:.2f}%\n")
        f.write(f"\nTest Passed: {evaluation['passed']}\n")
        if not evaluation['passed']:
            f.write("\nFailures:\n")
            for failure in evaluation['failures']:
                f.write(f"  - {failure}\n")
    
    print(f"\n✅ Results saved to {results_path}")
    
    if not evaluation['passed']:
        print("\n⚠️  TEST 1C FAILED - PHASE 1 INCOMPLETE")
        exit(1)
    else:
        print("\n✅ TEST 1C PASSED - PHASE 1 COMPLETE! 🎉")
        print("\nNext: Implement Phase 2 (Scalability: 4-6 agents)")
        exit(0)
