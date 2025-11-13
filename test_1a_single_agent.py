"""
TEST 1A: Single Agent, Empty Grid (FOUNDATION VALIDATION)

IMMUTABLE CONFIGURATION - DO NOT CHANGE
This test validates that basic Q-learning works in the simplest scenario.

Success Criteria (LOCKED):
  - Learned coverage: >75% (must beat random)
  - Random baseline: 50-60% (sanity check)
  - Loss stable: No divergence
  - Learning positive: Late > Early

If this fails, EVERYTHING else is pointless.
"""

import os
import time
import random
import numpy as np
import torch
from typing import Dict

from environment import MARL_QMIX_Environment
from utils import generate_empty_map, ensure_dir


# ============================================================
# LOCKED CONFIGURATION (DO NOT MODIFY)
# ============================================================
TEST_1A_CONFIG = {
    'grid_size': 10,
    'num_agents': 1,           # Single agent (no coordination needed)
    'sensor_range': 3,
    'comm_range': 0.0,          # N/A for single agent
    'coverage_threshold': 0.8,
    'completion_threshold_perc': 85.0,
    'max_episodes': 100,        # Sufficient for simple task
    'max_steps_per_episode': 50,
    
    # Rewards (standard)
    'gamma_coverage': 15.0,
    'step_penalty': -0.02,
    'orientation_cost_factor': 0.02,
    'invalid_move_penalty': -0.5,
    
    # Learning (Tier 1 stable config)
    'batch_size': 64,
    'memory_capacity': 10000,   # Smaller (single agent)
    'gamma': 0.99,
    'lr_agents': 5e-4,
    'lr_mixer': 1e-4,           # Not used (single agent)
    'use_soft_update': True,
    'soft_update_tau': 0.001,
    
    # Exploration (faster decay for 100 episodes)
    'agent_config': {
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay': 0.95,  # Reaches ~0.05 by episode 60
    },
    
    # Device
    'device': None,  # Auto-detect
    'tensorboard_dir': './runs/test_1a',
}

# ============================================================
# SUCCESS CRITERIA (LOCKED - DO NOT MODIFY)
# ============================================================
SUCCESS_CRITERIA_1A = {
    'learned_coverage_min': 75.0,       # Must beat random
    'random_baseline_min': 50.0,        # Sanity check lower bound
    'random_baseline_max': 60.0,        # Sanity check upper bound
    'loss_stable': True,                 # No explosion (max < 10.0)
    'learning_positive': True,           # Late > Early + 5%
    'episodes_to_convergence': 100,      # Must converge within budget
}


def run_random_baseline(num_episodes: int = 50, verbose: bool = False):
    """
    PURE RANDOM BASELINE - No learning, no networks.
    
    Expected: 50-60% coverage (random walk in empty 10×10 grid)
    """
    print("\n" + "="*70)
    print("RUNNING RANDOM BASELINE (PURE RANDOM)")
    print("="*70)
    
    # Same config but disable learning
    config = TEST_1A_CONFIG.copy()
    config['tensorboard_dir'] = None  # Disable logging
    
    env = MARL_QMIX_Environment(**config)
    
    # Force empty map
    env.training_map_generators['empty'] = lambda: generate_empty_map(10)
    env.training_map_order = ['empty']
    
    coverages = []
    
    for ep in range(num_episodes):
        env.reset(full_reset=(ep == 0), mode='train')
        
        for step in range(50):
            # PURE RANDOM: Select uniformly from valid actions
            actions = {}
            for agent_id in range(env.num_agents):
                valid_actions = env.get_valid_actions(agent_id)
                actions[agent_id] = random.choice(valid_actions) if valid_actions else (0, 0)
            
            # Execute (but don't learn)
            env.step(actions)
        
        final_coverage = env.calculate_coverage_percentage()
        coverages.append(final_coverage)
        
        if verbose and (ep + 1) % 10 == 0:
            recent_avg = np.mean(coverages[-10:])
            print(f"  Episode {ep+1}/{num_episodes}: Coverage={final_coverage:.1f}%, Avg(last 10)={recent_avg:.1f}%")
    
    mean_cov = np.mean(coverages)
    std_cov = np.std(coverages)
    
    print(f"\nRandom Baseline Results:")
    print(f"  Mean coverage: {mean_cov:.2f}% ± {std_cov:.2f}%")
    print(f"  Range: [{np.min(coverages):.1f}%, {np.max(coverages):.1f}%]")
    print(f"  Expected: 50-60% (sanity check)")
    
    return {
        'mean_coverage': mean_cov,
        'std_coverage': std_cov,
        'coverages': coverages,
    }


def run_learned_policy(num_episodes: int = 100, verbose: bool = True):
    """
    LEARNED POLICY - Train single agent with Q-learning.
    
    Expected: >75% coverage (beats random)
    """
    print("\n" + "="*70)
    print("RUNNING LEARNED POLICY (Q-LEARNING)")
    print("="*70)
    
    env = MARL_QMIX_Environment(**TEST_1A_CONFIG)
    
    # Force empty map
    env.training_map_generators['empty'] = lambda: generate_empty_map(10)
    env.training_map_order = ['empty']
    
    # Reset environment
    env.reset(full_reset=True, mode='train')
    
    coverages = []
    losses = []
    
    print(f"Training for {num_episodes} episodes...")
    print(f"  Grid: 10×10 empty")
    print(f"  Agent: 1 (no coordination)")
    print(f"  Steps: 50 per episode")
    
    for ep in range(num_episodes):
        env.reset(full_reset=False, mode='train')
        
        episode_losses = []
        
        for step in range(50):
            # Agent selects action (epsilon-greedy)
            actions = {}
            for agent_id, agent in enumerate(env.agents):
                state_tensor = agent.get_state_tensor()
                valid_actions = env.get_valid_actions(agent_id)
                action = agent.select_action(state_tensor, valid_actions)
                actions[agent_id] = action
            
            # Execute and learn
            results = env.step(actions)
            
            # Track loss
            if env.metrics.qmix_loss:
                episode_losses.append(env.metrics.qmix_loss[-1])
        
        final_coverage = env.calculate_coverage_percentage()
        coverages.append(final_coverage)
        
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        # Decay epsilon after each episode
        for agent in env.agents:
            agent.update_epsilon()
        
        if verbose and (ep + 1) % 10 == 0:
            recent_cov = np.mean(coverages[-10:])
            recent_loss = np.mean(losses[-10:]) if losses else 0.0
            epsilon = env.agents[0].epsilon
            print(f"  Episode {ep+1}/{num_episodes}: "
                  f"Coverage={final_coverage:.1f}%, "
                  f"Avg(last 10)={recent_cov:.1f}%, "
                  f"Loss={recent_loss:.4f}, "
                  f"ε={epsilon:.3f}")
    
    # Compute statistics
    early_cov = np.mean(coverages[:20]) if len(coverages) >= 20 else np.mean(coverages[:10])
    late_cov = np.mean(coverages[-20:]) if len(coverages) >= 20 else np.mean(coverages[-10:])
    improvement = late_cov - early_cov
    
    mean_cov = np.mean(coverages)
    std_cov = np.std(coverages)
    
    mean_loss = np.mean(losses) if losses else 0.0
    max_loss = np.max(losses) if losses else 0.0
    
    print(f"\nLearned Policy Results:")
    print(f"  Coverage (early):  {early_cov:.2f}%")
    print(f"  Coverage (late):   {late_cov:.2f}%")
    print(f"  Improvement:       {improvement:+.2f}%")
    print(f"  Mean coverage:     {mean_cov:.2f}% ± {std_cov:.2f}%")
    print(f"  Loss (mean):       {mean_loss:.4f}")
    print(f"  Loss (max):        {max_loss:.4f}")
    print(f"  Expected:          >75% coverage")
    
    return {
        'mean_coverage': mean_cov,
        'std_coverage': std_cov,
        'early_coverage': early_cov,
        'late_coverage': late_cov,
        'improvement': improvement,
        'coverages': coverages,
        'losses': losses,
        'mean_loss': mean_loss,
        'max_loss': max_loss,
    }


def evaluate_test_1a(random_results: Dict, learned_results: Dict) -> Dict:
    """
    Evaluate Test 1A against LOCKED success criteria.
    
    Returns pass/fail with detailed diagnostics.
    """
    print("\n" + "="*70)
    print("TEST 1A EVALUATION")
    print("="*70)
    
    criteria = SUCCESS_CRITERIA_1A
    results = {
        'test': 'Test_1A',
        'config': 'Single agent, empty 10×10 grid',
        'passed': True,
        'failures': [],
    }
    
    # Check 1: Random baseline in expected range
    random_mean = random_results['mean_coverage']
    if random_mean < criteria['random_baseline_min'] or random_mean > criteria['random_baseline_max']:
        results['passed'] = False
        results['failures'].append(
            f"Random baseline {random_mean:.1f}% outside expected range "
            f"[{criteria['random_baseline_min']}, {criteria['random_baseline_max']}]"
        )
        print(f"❌ FAIL: Random baseline sanity check")
    else:
        print(f"✅ PASS: Random baseline {random_mean:.1f}% in range")
    
    # Check 2: Learned policy beats threshold
    learned_mean = learned_results['mean_coverage']
    if learned_mean < criteria['learned_coverage_min']:
        results['passed'] = False
        results['failures'].append(
            f"Learned coverage {learned_mean:.1f}% below threshold {criteria['learned_coverage_min']}"
        )
        print(f"❌ FAIL: Learned coverage too low")
    else:
        print(f"✅ PASS: Learned coverage {learned_mean:.1f}% > {criteria['learned_coverage_min']}%")
    
    # Check 3: Learned beats random
    if learned_mean <= random_mean:
        results['passed'] = False
        results['failures'].append(
            f"Learned {learned_mean:.1f}% NOT better than random {random_mean:.1f}%"
        )
        print(f"❌ FAIL: Learning did not improve over random")
    else:
        gap = learned_mean - random_mean
        print(f"✅ PASS: Learned beats random by {gap:.1f}%")
    
    # Check 4: Loss stability
    max_loss = learned_results['max_loss']
    if max_loss > 10.0:
        results['passed'] = False
        results['failures'].append(f"Loss unstable: max={max_loss:.2f} (threshold 10.0)")
        print(f"❌ FAIL: Loss diverged (max={max_loss:.2f})")
    else:
        print(f"✅ PASS: Loss stable (max={max_loss:.2f})")
    
    # Check 5: Positive learning
    improvement = learned_results['improvement']
    if improvement < 5.0:
        results['passed'] = False
        results['failures'].append(f"Learning improvement {improvement:.1f}% < 5%")
        print(f"❌ FAIL: Insufficient learning progress")
    else:
        print(f"✅ PASS: Learning improved by {improvement:.1f}%")
    
    # Final verdict
    print("\n" + "="*70)
    if results['passed']:
        print("✅✅✅ TEST 1A PASSED ✅✅✅")
        print("FOUNDATION VALIDATED - PROCEED TO TEST 1B")
    else:
        print("❌❌❌ TEST 1A FAILED ❌❌❌")
        print("DO NOT PROCEED - FIX FAILURES FIRST")
        print("\nFailures:")
        for failure in results['failures']:
            print(f"  - {failure}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("\n" + "#"*70)
    print("# TEST 1A: SINGLE AGENT, EMPTY GRID")
    print("# FOUNDATION VALIDATION - IMMUTABLE")
    print("#"*70)
    print("\nPurpose: Validate that basic Q-learning works")
    print("Config: 1 agent, 10×10 empty grid, 100 episodes")
    print("Success: Learned >75%, beats random by >15%")
    print("\n" + "#"*70)
    
    # Step 1: Random baseline
    random_results = run_random_baseline(num_episodes=50, verbose=True)
    
    # Step 2: Learned policy
    learned_results = run_learned_policy(num_episodes=100, verbose=True)
    
    # Step 3: Evaluate
    evaluation = evaluate_test_1a(random_results, learned_results)
    
    # Save results
    ensure_dir('./test_results')
    results_path = './test_results/test_1a_results.txt'
    with open(results_path, 'w') as f:
        f.write("TEST 1A RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Random Baseline: {random_results['mean_coverage']:.2f}% ± {random_results['std_coverage']:.2f}%\n")
        f.write(f"Learned Policy:  {learned_results['mean_coverage']:.2f}% ± {learned_results['std_coverage']:.2f}%\n")
        f.write(f"Improvement:     {learned_results['improvement']:+.2f}%\n")
        f.write(f"Loss (max):      {learned_results['max_loss']:.4f}\n")
        f.write(f"\nTest Passed: {evaluation['passed']}\n")
        if not evaluation['passed']:
            f.write("\nFailures:\n")
            for failure in evaluation['failures']:
                f.write(f"  - {failure}\n")
    
    print(f"\n✅ Results saved to {results_path}")
    
    if not evaluation['passed']:
        print("\n⚠️  TEST 1A FAILED - DO NOT PROCEED TO TEST 1B")
        exit(1)
    else:
        print("\n✅ TEST 1A PASSED - READY FOR TEST 1B")
        exit(0)
