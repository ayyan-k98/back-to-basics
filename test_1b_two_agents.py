"""
TEST 1B: Two Agents, Empty Grid (COORDINATION VALIDATION)

IMMUTABLE CONFIGURATION - DO NOT CHANGE
This test validates that QMIX coordination helps over independent learning.

Success Criteria (LOCKED):
  - QMIX coverage: >80%
  - Independent coverage: 70-80%
  - QMIX beats Independent: p<0.05 (statistical significance)
  - Random baseline: 50-60%
  - Greedy baseline: 65-75%
  - No negative learning: Late ≥ Early

Prerequisite: Test 1A must PASS first
If Test 1B fails, coordination is NOT helping.
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
from utils import generate_empty_map, ensure_dir


# ============================================================
# LOCKED CONFIGURATION (DO NOT MODIFY)
# ============================================================
TEST_1B_CONFIG = {
    'grid_size': 15,
    'num_agents': 2,            # Two agents (coordination possible)
    'sensor_range': 3,
    'comm_range': 6.0,          # Can communicate within range
    'coverage_threshold': 0.8,
    'completion_threshold_perc': 85.0,
    'max_episodes': 200,        # More episodes for coordination learning
    'max_steps_per_episode': 100,
    
    # Rewards (standard)
    'gamma_coverage': 15.0,
    'step_penalty': -0.02,
    'orientation_cost_factor': 0.02,
    'invalid_move_penalty': -0.5,
    
    # Learning (Tier 1 stable config)
    'batch_size': 64,
    'memory_capacity': 20000,
    'gamma': 0.99,
    'lr_agents': 5e-4,          # Agent networks
    'lr_mixer': 1e-4,           # Mixer 5x slower
    'use_soft_update': True,
    'soft_update_tau': 0.001,
    
    # Exploration (faster decay for 200 episodes)
    'agent_config': {
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.97,  # Reaches ~0.05 by episode 100
    },
    
    # Device
    'device': None,  # Auto-detect
    'tensorboard_dir': './runs/test_1b',
}

# ============================================================
# SUCCESS CRITERIA (LOCKED - DO NOT MODIFY)
# ============================================================
SUCCESS_CRITERIA_1B = {
    'qmix_coverage_min': 80.0,          # QMIX must achieve this
    'independent_coverage_min': 70.0,   # Independent lower bound
    'independent_coverage_max': 80.0,   # Independent upper bound
    'qmix_beats_independent': True,     # QMIX > Independent
    'statistical_significance': 0.05,   # p < 0.05
    'random_baseline_min': 50.0,
    'random_baseline_max': 60.0,
    'greedy_baseline_min': 65.0,
    'greedy_baseline_max': 75.0,
    'no_negative_learning': True,       # Late ≥ Early
    'loss_stable': True,                # Max loss < 10.0
}


def run_random_baseline_2agents(num_episodes: int = 50, verbose: bool = False):
    """
    PURE RANDOM BASELINE - 2 agents, no learning.
    
    Expected: 50-60% coverage
    """
    print("\n" + "="*70)
    print("RUNNING RANDOM BASELINE (2 AGENTS)")
    print("="*70)
    
    config = TEST_1B_CONFIG.copy()
    config['tensorboard_dir'] = None
    
    env = MARL_QMIX_Environment(**config)
    env.training_map_generators['empty'] = lambda: generate_empty_map(15)
    env.training_map_order = ['empty']
    
    coverages = []
    
    for ep in range(num_episodes):
        env.reset(full_reset=(ep == 0), mode='train')
        
        for step in range(100):
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


def run_greedy_baseline_2agents(num_episodes: int = 50, verbose: bool = False):
    """
    GREEDY FRONTIER BASELINE - 2 agents, heuristic only.
    
    Expected: 65-75% coverage
    """
    print("\n" + "="*70)
    print("RUNNING GREEDY BASELINE (2 AGENTS)")
    print("="*70)
    
    config = TEST_1B_CONFIG.copy()
    config['tensorboard_dir'] = None
    
    env = MARL_QMIX_Environment(**config)
    env.training_map_generators['empty'] = lambda: generate_empty_map(15)
    env.training_map_order = ['empty']
    
    env.reset(full_reset=True, mode='train')
    baseline_runner = BaselineRunner(env, baseline_type='greedy')
    
    coverages = []
    
    for ep in range(num_episodes):
        env.reset(full_reset=False, mode='train')
        
        for step in range(100):
            actions = baseline_runner.select_actions()
            env.step(actions)
        
        final_coverage = env.calculate_coverage_percentage()
        coverages.append(final_coverage)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}: Coverage={final_coverage:.1f}%, Avg={np.mean(coverages):.1f}%")
    
    mean_cov = np.mean(coverages)
    print(f"\nGreedy: {mean_cov:.2f}% ± {np.std(coverages):.2f}%")
    
    return {'mean_coverage': mean_cov, 'std_coverage': np.std(coverages), 'coverages': coverages}


def run_independent_ql_2agents(num_episodes: int = 200, verbose: bool = True):
    """
    INDEPENDENT Q-LEARNING - 2 agents, no coordination.
    
    Expected: 70-80% coverage
    """
    print("\n" + "="*70)
    print("RUNNING INDEPENDENT Q-LEARNING (2 AGENTS)")
    print("="*70)
    
    config = TEST_1B_CONFIG.copy()
    config['tensorboard_dir'] = None
    
    env = MARL_QMIX_Environment(**config)
    env.training_map_generators['empty'] = lambda: generate_empty_map(15)
    env.training_map_order = ['empty']
    
    env.reset(full_reset=True, mode='train')
    baseline_runner = BaselineRunner(env, baseline_type='independent')
    
    coverages = []
    losses = []
    
    print(f"Training for {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        env.reset(full_reset=False, mode='train')
        
        episode_losses = []
        
        for step in range(100):
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
        
        if verbose and (ep + 1) % 20 == 0:
            recent_cov = np.mean(coverages[-20:])
            recent_loss = np.mean(losses[-20:]) if losses else 0.0
            epsilon = baseline_runner.get_epsilon()
            print(f"  Episode {ep+1}: Coverage={final_coverage:.1f}%, "
                  f"Avg(last 20)={recent_cov:.1f}%, Loss={recent_loss:.4f}, ε={epsilon:.3f}")
    
    early_cov = np.mean(coverages[:30])
    late_cov = np.mean(coverages[-30:])
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


def run_qmix_2agents(num_episodes: int = 200, verbose: bool = True):
    """
    QMIX - 2 agents with coordination (mixer network).
    
    Expected: >80% coverage, beats Independent
    """
    print("\n" + "="*70)
    print("RUNNING QMIX (2 AGENTS WITH COORDINATION)")
    print("="*70)
    
    env = MARL_QMIX_Environment(**TEST_1B_CONFIG)
    env.training_map_generators['empty'] = lambda: generate_empty_map(15)
    env.training_map_order = ['empty']
    
    env.reset(full_reset=True, mode='train')
    
    coverages = []
    losses = []
    
    print(f"Training for {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        env.reset(full_reset=False, mode='train')
        
        for step in range(100):
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
            recent_losses = env.metrics.qmix_loss[-100:]
            losses.append(np.mean(recent_losses))
        
        # Decay epsilon after each episode
        for agent in env.agents:
            agent.update_epsilon()
        
        if verbose and (ep + 1) % 20 == 0:
            recent_cov = np.mean(coverages[-20:])
            recent_loss = np.mean(losses[-20:]) if losses else 0.0
            epsilon = env.agents[0].epsilon
            print(f"  Episode {ep+1}: Coverage={final_coverage:.1f}%, "
                  f"Avg(last 20)={recent_cov:.1f}%, Loss={recent_loss:.4f}, ε={epsilon:.3f}")
    
    early_cov = np.mean(coverages[:30])
    late_cov = np.mean(coverages[-30:])
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


def evaluate_test_1b(random_results, greedy_results, independent_results, qmix_results):
    """
    Evaluate Test 1B against LOCKED success criteria.
    """
    print("\n" + "="*70)
    print("TEST 1B EVALUATION")
    print("="*70)
    
    criteria = SUCCESS_CRITERIA_1B
    results = {
        'test': 'Test_1B',
        'config': 'Two agents, empty 15×15 grid',
        'passed': True,
        'failures': [],
    }
    
    # Check 1: Random baseline sanity
    random_mean = random_results['mean_coverage']
    if not (criteria['random_baseline_min'] <= random_mean <= criteria['random_baseline_max']):
        results['passed'] = False
        results['failures'].append(f"Random {random_mean:.1f}% outside [50, 60]")
        print(f"❌ FAIL: Random baseline sanity check")
    else:
        print(f"✅ PASS: Random {random_mean:.1f}% in range")
    
    # Check 2: Greedy baseline sanity
    greedy_mean = greedy_results['mean_coverage']
    if not (criteria['greedy_baseline_min'] <= greedy_mean <= criteria['greedy_baseline_max']):
        results['passed'] = False
        results['failures'].append(f"Greedy {greedy_mean:.1f}% outside [65, 75]")
        print(f"❌ FAIL: Greedy baseline sanity check")
    else:
        print(f"✅ PASS: Greedy {greedy_mean:.1f}% in range")
    
    # Check 3: Independent coverage
    indep_mean = independent_results['mean_coverage']
    if not (criteria['independent_coverage_min'] <= indep_mean <= criteria['independent_coverage_max']):
        results['passed'] = False
        results['failures'].append(f"Independent {indep_mean:.1f}% outside [70, 80]")
        print(f"❌ FAIL: Independent coverage out of range")
    else:
        print(f"✅ PASS: Independent {indep_mean:.1f}% in range")
    
    # Check 4: QMIX coverage threshold
    qmix_mean = qmix_results['mean_coverage']
    if qmix_mean < criteria['qmix_coverage_min']:
        results['passed'] = False
        results['failures'].append(f"QMIX {qmix_mean:.1f}% < 80%")
        print(f"❌ FAIL: QMIX coverage too low")
    else:
        print(f"✅ PASS: QMIX {qmix_mean:.1f}% > 80%")
    
    # Check 5: QMIX beats Independent
    if qmix_mean <= indep_mean:
        results['passed'] = False
        results['failures'].append(f"QMIX {qmix_mean:.1f}% NOT better than Independent {indep_mean:.1f}%")
        print(f"❌ FAIL: QMIX doesn't beat Independent")
    else:
        gap = qmix_mean - indep_mean
        print(f"✅ PASS: QMIX beats Independent by {gap:.1f}%")
    
    # Check 6: Statistical significance
    t_stat, p_value = stats.ttest_ind(qmix_results['coverages'], independent_results['coverages'])
    if p_value >= criteria['statistical_significance']:
        results['passed'] = False
        results['failures'].append(f"Not statistically significant: p={p_value:.4f} >= 0.05")
        print(f"❌ FAIL: Not statistically significant (p={p_value:.4f})")
    else:
        print(f"✅ PASS: Statistically significant (p={p_value:.4f})")
    
    # Check 7: No negative learning (QMIX)
    if qmix_results['improvement'] < 0:
        results['passed'] = False
        results['failures'].append(f"QMIX negative learning: {qmix_results['improvement']:.1f}%")
        print(f"❌ FAIL: QMIX shows negative learning")
    else:
        print(f"✅ PASS: QMIX positive learning ({qmix_results['improvement']:+.1f}%)")
    
    # Final verdict
    print("\n" + "="*70)
    if results['passed']:
        print("✅✅✅ TEST 1B PASSED ✅✅✅")
        print("COORDINATION VALIDATED - PROCEED TO TEST 1C")
    else:
        print("❌❌❌ TEST 1B FAILED ❌❌❌")
        print("DO NOT PROCEED - FIX COORDINATION MECHANISM")
        print("\nFailures:")
        for failure in results['failures']:
            print(f"  - {failure}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    # Check Test 1A passed
    if not os.path.exists('./test_results/test_1a_results.txt'):
        print("❌ ERROR: Test 1A must be run first")
        print("Run: py test_1a_single_agent.py")
        exit(1)
    
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("\n" + "#"*70)
    print("# TEST 1B: TWO AGENTS, EMPTY GRID")
    print("# COORDINATION VALIDATION - IMMUTABLE")
    print("#"*70)
    print("\nPurpose: Validate that QMIX coordination helps")
    print("Config: 2 agents, 15×15 empty grid, 200 episodes")
    print("Success: QMIX >80%, beats Independent (p<0.05)")
    print("\n" + "#"*70)
    
    # Step 1: Random baseline
    random_results = run_random_baseline_2agents(num_episodes=50, verbose=True)
    
    # Step 2: Greedy baseline
    greedy_results = run_greedy_baseline_2agents(num_episodes=50, verbose=True)
    
    # Step 3: Independent Q-Learning
    independent_results = run_independent_ql_2agents(num_episodes=200, verbose=True)
    
    # Step 4: QMIX
    qmix_results = run_qmix_2agents(num_episodes=200, verbose=True)
    
    # Step 5: Evaluate
    evaluation = evaluate_test_1b(random_results, greedy_results, independent_results, qmix_results)
    
    # Save results
    ensure_dir('./test_results')
    results_path = './test_results/test_1b_results.txt'
    with open(results_path, 'w') as f:
        f.write("TEST 1B RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Random:       {random_results['mean_coverage']:.2f}%\n")
        f.write(f"Greedy:       {greedy_results['mean_coverage']:.2f}%\n")
        f.write(f"Independent:  {independent_results['mean_coverage']:.2f}%\n")
        f.write(f"QMIX:         {qmix_results['mean_coverage']:.2f}%\n")
        f.write(f"\nGap (QMIX - Independent): {qmix_results['mean_coverage'] - independent_results['mean_coverage']:.2f}%\n")
        f.write(f"\nTest Passed: {evaluation['passed']}\n")
        if not evaluation['passed']:
            f.write("\nFailures:\n")
            for failure in evaluation['failures']:
                f.write(f"  - {failure}\n")
    
    print(f"\n✅ Results saved to {results_path}")
    
    if not evaluation['passed']:
        print("\n⚠️  TEST 1B FAILED - DO NOT PROCEED TO TEST 1C")
        exit(1)
    else:
        print("\n✅ TEST 1B PASSED - READY FOR TEST 1C")
        exit(0)
