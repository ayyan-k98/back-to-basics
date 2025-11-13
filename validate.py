"""
Minimal Validation Suite: Does QMIX Actually Learn?

Tests basic learning on toy problems before adding ANY features.
If these fail, we have bugs. If they pass, we have a baseline.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from environment import MARL_QMIX_Environment
from train import train
from utils import ensure_dir


def test_2agents_empty_5x5():
    """
    Test 1: CRITICAL - Can 2 agents learn to cover empty 5x5 grid?

    Expected: Should reach 100% coverage within 50 episodes.
    If fails: QMIX implementation is broken.
    """
    print("\n" + "="*70)
    print("TEST 1: 2 Agents, Empty 5x5 Grid")
    print("="*70)
    print("Expected: >95% coverage within 50 episodes")
    print("This is the SIMPLEST possible test. If this fails, QMIX is broken.\n")

    env = MARL_QMIX_Environment(
        grid_size=5,
        num_agents=2,
        sensor_range=3,
        comm_range=10.0,
        coverage_threshold=0.9,
        completion_threshold_perc=95.0,
        max_episodes=50,
        max_steps_per_episode=15,
        device="cpu",
        tensorboard_dir="./runs/validation_test1",
        # QMIX params
        batch_size=32,
        memory_capacity=5000,
        gamma=0.99,
        lr=0.001,
        # Agent params
        agent_config={
            'epsilon_start': 1.0,
            'epsilon_end': 0.1,
            'epsilon_decay': 0.95,
        },
        # Reward params
        gamma_coverage=30.0,
        step_penalty=-0.05,
    )

    # Override map to be EMPTY (no obstacles)
    def empty_map_generator():
        grid = np.zeros((5, 5), dtype=np.float32)
        grid[0, :] = 1.0
        grid[-1, :] = 1.0
        grid[:, 0] = 1.0
        grid[:, -1] = 1.0
        return grid

    env.training_map_generators['empty'] = empty_map_generator
    env.training_map_order = ['empty']

    print("Training...")
    metrics, _, _, _, _ = train(
        env,
        episodes_per_type=50,
        verbose=False,
        save_interval=50,
        model_dir="./models_validation_test1"
    )

    # Analyze results
    final_coverage = env.calculate_coverage_percentage()
    max_coverage = max([env.calculate_coverage_percentage()
                       for _ in range(10)])  # Sample 10 episodes

    print("\n" + "-"*70)
    print("RESULTS:")
    print(f"  Final coverage: {final_coverage:.1f}%")
    print(f"  Max coverage: {max_coverage:.1f}%")
    print(f"  Episodes completed: {len(metrics.episode_coverage)}")

    # Check if learning happened (using EPISODE-level metrics now)
    early_coverage = np.mean(metrics.episode_coverage[:10]) if len(metrics.episode_coverage) >= 10 else 0
    late_coverage = np.mean(metrics.episode_coverage[-10:]) if len(metrics.episode_coverage) >= 10 else 0
    improvement = late_coverage - early_coverage

    print(f"  Early coverage (eps 1-10): {early_coverage:.2f}%")
    print(f"  Late coverage (eps 40-50): {late_coverage:.2f}%")
    print(f"  Improvement: {improvement:.2f}%")

    # Check if QMIX loss decreased (using episode-level average loss)
    if metrics.episode_avg_loss:
        early_loss = np.mean(metrics.episode_avg_loss[:10]) if len(metrics.episode_avg_loss) >= 10 else 0
        late_loss = np.mean(metrics.episode_avg_loss[-10:]) if len(metrics.episode_avg_loss) >= 10 else 0
        print(f"  Early loss (eps 1-10): {early_loss:.4f}")
        print(f"  Late loss (eps 40-50): {late_loss:.4f}")
        print(f"  Loss change: {late_loss - early_loss:.4f}")

    # Pass/Fail
    passed = max_coverage > 95.0 and improvement > 0
    print("-"*70)
    if passed:
        print("✅ TEST 1 PASSED: QMIX learns on simple 5x5 grid")
    else:
        print("❌ TEST 1 FAILED: QMIX does not learn basic coverage")
        print("   FIX THIS BEFORE PROCEEDING!")
    print("="*70)

    return passed, metrics


def test_4agents_empty_10x10():
    """
    Test 2: IMPORTANT - Does it scale to 4 agents, 10x10?

    Expected: Should reach >90% coverage within 200 episodes.
    If fails: Scaling/coordination issues.
    """
    print("\n" + "="*70)
    print("TEST 2: 4 Agents, Empty 10x10 Grid")
    print("="*70)
    print("Expected: >90% coverage within 200 episodes")
    print("Tests if coordination works at moderate scale.\n")

    env = MARL_QMIX_Environment(
        grid_size=10,
        num_agents=4,
        sensor_range=4,
        comm_range=8.0,
        coverage_threshold=0.8,
        completion_threshold_perc=90.0,
        max_episodes=200,
        max_steps_per_episode=100,
        device="cpu",
        tensorboard_dir="./runs/validation_test2",
        batch_size=64,
        memory_capacity=10000,
        gamma=0.99,
        lr=0.0005,
        agent_config={
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.98,
        },
        gamma_coverage=20.0,
        step_penalty=-0.02,
    )

    # Empty grid
    def empty_map_generator():
        grid = np.zeros((10, 10), dtype=np.float32)
        grid[0, :] = 1.0
        grid[-1, :] = 1.0
        grid[:, 0] = 1.0
        grid[:, -1] = 1.0
        return grid

    env.training_map_generators['empty'] = empty_map_generator
    env.training_map_order = ['empty']

    print("Training (this may take a few minutes)...")
    metrics, _, _, _, _ = train(
        env,
        episodes_per_type=200,
        verbose=False,
        save_interval=200,
        model_dir="./models_validation_test2"
    )

    # Analyze
    final_coverage = env.calculate_coverage_percentage()

    early_coverage = np.mean(metrics.episode_coverage[:20]) if len(metrics.episode_coverage) >= 20 else 0
    late_coverage = np.mean(metrics.episode_coverage[-20:]) if len(metrics.episode_coverage) >= 20 else 0
    improvement = late_coverage - early_coverage

    print("\n" + "-"*70)
    print("RESULTS:")
    print(f"  Final coverage: {final_coverage:.1f}%")
    print(f"  Episodes completed: {len(metrics.episode_coverage)}")
    print(f"  Early coverage (eps 1-20): {early_coverage:.2f}%")
    print(f"  Late coverage (eps 180-200): {late_coverage:.2f}%")
    print(f"  Improvement: {improvement:.2f}%")

    if metrics.episode_avg_loss:
        early_loss = np.mean(metrics.episode_avg_loss[:20]) if len(metrics.episode_avg_loss) >= 20 else 0
        late_loss = np.mean(metrics.episode_avg_loss[-20:]) if len(metrics.episode_avg_loss) >= 20 else 0
        print(f"  Early loss (eps 1-20): {early_loss:.4f}")
        print(f"  Late loss (eps 180-200): {late_loss:.4f}")
        print(f"  Loss change: {late_loss - early_loss:.4f}")

    passed = final_coverage > 85.0 and improvement > 0
    print("-"*70)
    if passed:
        print("✅ TEST 2 PASSED: QMIX scales to 4 agents on 10x10")
    else:
        print("❌ TEST 2 FAILED: Scaling or coordination problems")
    print("="*70)

    return passed, metrics


def test_2agents_obstacles_10x10():
    """
    Test 3: VALIDATION - Can agents handle obstacles?

    Expected: Should reach >80% coverage within 300 episodes.
    If fails: POMDP or raycasting issues.
    """
    print("\n" + "="*70)
    print("TEST 3: 2 Agents, 10x10 Grid with Obstacles")
    print("="*70)
    print("Expected: >80% coverage within 300 episodes")
    print("Tests POMDP integrity and obstacle handling.\n")

    from utils import generate_room_map

    env = MARL_QMIX_Environment(
        grid_size=10,
        num_agents=2,
        sensor_range=4,
        comm_range=8.0,
        coverage_threshold=0.8,
        completion_threshold_perc=80.0,
        max_episodes=300,
        max_steps_per_episode=100,
        device="cpu",
        tensorboard_dir="./runs/validation_test3",
        batch_size=64,
        memory_capacity=15000,
        gamma=0.99,
        lr=0.0005,
        agent_config={
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.99,
        },
        gamma_coverage=25.0,
        step_penalty=-0.02,
    )

    # Simple room map
    def simple_room_generator():
        return generate_room_map(10, num_rooms_range=(2, 3), room_size_range=(3, 4))

    env.training_map_generators['room'] = simple_room_generator
    env.training_map_order = ['room']

    print("Training (this may take several minutes)...")
    metrics, _, _, _, _ = train(
        env,
        episodes_per_type=300,
        verbose=False,
        save_interval=300,
        model_dir="./models_validation_test3"
    )

    # Analyze
    final_coverage = env.calculate_coverage_percentage()

    early_coverage = np.mean(metrics.episode_coverage[:30]) if len(metrics.episode_coverage) >= 30 else 0
    late_coverage = np.mean(metrics.episode_coverage[-30:]) if len(metrics.episode_coverage) >= 30 else 0
    improvement = late_coverage - early_coverage

    print("\n" + "-"*70)
    print("RESULTS:")
    print(f"  Final coverage: {final_coverage:.1f}%")
    print(f"  Episodes completed: {len(metrics.episode_coverage)}")
    print(f"  Early coverage (eps 1-30): {early_coverage:.2f}%")
    print(f"  Late coverage (eps 270-300): {late_coverage:.2f}%")
    print(f"  Improvement: {improvement:.2f}%")

    if metrics.episode_avg_loss:
        early_loss = np.mean(metrics.episode_avg_loss[:30]) if len(metrics.episode_avg_loss) >= 30 else 0
        late_loss = np.mean(metrics.episode_avg_loss[-30:]) if len(metrics.episode_avg_loss) >= 30 else 0
        print(f"  Early loss (eps 1-30): {early_loss:.4f}")
        print(f"  Late loss (eps 270-300): {late_loss:.4f}")
        print(f"  Loss change: {late_loss - early_loss:.4f}")

    passed = final_coverage > 75.0 and improvement > 0
    print("-"*70)
    if passed:
        print("✅ TEST 3 PASSED: QMIX handles obstacles")
    else:
        print("❌ TEST 3 FAILED: POMDP or obstacle handling issues")
    print("="*70)

    return passed, metrics


def plot_validation_results(test1_metrics, test2_metrics, test3_metrics):
    """Plot comparison of all 3 validation tests."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('QMIX Validation Results', fontsize=16)

    tests = [
        ("Test 1: 2 agents, 5x5", test1_metrics),
        ("Test 2: 4 agents, 10x10", test2_metrics),
        ("Test 3: 2 agents, 10x10 obstacles", test3_metrics)
    ]

    for col, (title, metrics) in enumerate(tests):
        # Coverage over time (episode-level)
        ax = axes[0, col]
        if metrics and metrics.episode_coverage:
            ax.plot(metrics.episode_coverage, alpha=0.7)
            ax.set_title(f'{title}\nCoverage Progress')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Coverage %')
            ax.grid(True, alpha=0.3)

        # Loss over time (episode-level average)
        ax = axes[1, col]
        if metrics and metrics.episode_avg_loss:
            ax.plot(metrics.episode_avg_loss, 'r-', alpha=0.7)
            ax.set_title('QMIX Loss (per episode avg)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ensure_dir('./validation_results')
    plt.savefig('./validation_results/validation_comparison.png', dpi=150)
    print("\n📊 Validation plots saved to: ./validation_results/validation_comparison.png")
    plt.close()


def run_validation_suite():
    """
    Run all validation tests in sequence.

    This tells us if QMIX works AT ALL before adding any features.
    """
    print("\n" + "="*70)
    print("QMIX VALIDATION SUITE")
    print("="*70)
    print("Purpose: Verify QMIX learns BEFORE adding features")
    print("="*70)

    results = []

    # Test 1: Simplest case
    try:
        passed1, metrics1 = test_2agents_empty_5x5()
        results.append(("Test 1: 2 agents, 5x5 empty", passed1))

        if not passed1:
            print("\n⚠️  WARNING: Test 1 failed!")
            print("Fix basic QMIX implementation before proceeding to Test 2.")
            return False
    except Exception as e:
        print(f"\n❌ Test 1 CRASHED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Scaling
    try:
        passed2, metrics2 = test_4agents_empty_10x10()
        results.append(("Test 2: 4 agents, 10x10 empty", passed2))
    except Exception as e:
        print(f"\n❌ Test 2 CRASHED: {e}")
        metrics2 = None
        passed2 = False
        results.append(("Test 2: 4 agents, 10x10 empty", False))

    # Test 3: Obstacles
    try:
        passed3, metrics3 = test_2agents_obstacles_10x10()
        results.append(("Test 3: 2 agents, 10x10 obstacles", passed3))
    except Exception as e:
        print(f"\n❌ Test 3 CRASHED: {e}")
        metrics3 = None
        passed3 = False
        results.append(("Test 3: 2 agents, 10x10 obstacles", False))

    # Generate comparison plots
    try:
        plot_validation_results(metrics1, metrics2, metrics3)
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    print("="*70)

    if passed_tests == total_tests:
        print("\n🎉 ALL VALIDATION TESTS PASSED!")
        print("Your QMIX implementation is working.")
        print("\nNext steps:")
        print("1. Implement baselines (greedy, independent Q-learning)")
        print("2. Run comparison experiments")
        print("3. Analyze where QMIX struggles")
        print("4. Add features ONLY if needed based on analysis")
    elif passed_tests >= 1:
        print("\n⚠️  PARTIAL SUCCESS")
        print("Some tests passed, some failed.")
        print("Debug the failures before adding features.")
    else:
        print("\n❌ ALL TESTS FAILED")
        print("QMIX implementation has fundamental issues.")
        print("DO NOT add features. Fix the basics first.")

    return passed_tests == total_tests


if __name__ == "__main__":
    # Set seeds for reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    success = run_validation_suite()

    if success:
        print("\n✅ Validation complete. Ready for baseline comparisons.")
    else:
        print("\n❌ Validation failed. Fix implementation before proceeding.")
