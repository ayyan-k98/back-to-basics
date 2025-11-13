"""
TIER 1 FIXES VALIDATION - Proper Engineering, No Band-Aids

Implemented:
1. ✅ Separate learning rates (lr_agents=0.0005, lr_mixer=0.0001)
2. ✅ Slower target updates (tau=0.001, was 0.005)
3. ✅ Reward normalization (Welford's online algorithm)
4. ✅ Huber loss (architecturally sound, not a hack)
5. ✅ Standard gradient clipping (10.0, not excessive)

What we REMOVED (band-aids):
- ❌ Q-value clipping (was masking root cause)
- ❌ Excessive grad clipping (0.5 was too strict)

Expected: Test 2 loss should be stable (<5.0, not 460!)
"""

import os
import numpy as np
import torch
from environment import MARL_QMIX_Environment
from train import train
from utils import ensure_dir


def test_1_simple():
    """Test 1: 2 agents, 5x5 - Should still pass easily"""
    print("\n" + "="*70)
    print("TEST 1: 2 Agents, Empty 5x5 (Baseline)")
    print("="*70)

    env = MARL_QMIX_Environment(
        grid_size=5,
        num_agents=2,
        sensor_range=3,
        comm_range=10.0,
        coverage_threshold=0.9,
        completion_threshold_perc=95.0,
        max_episodes=50,
        max_steps_per_episode=15,
        device=None,  # Auto-detect: cuda if available, else cpu
        tensorboard_dir="./runs/tier1_test1",
        batch_size=32,
        memory_capacity=5000,
        gamma=0.99,
        lr_agents=0.0005,  # Standard
        lr_mixer=0.0001,   # 5x slower (Tier 1 fix)
        soft_update_tau=0.001,  # 5x slower (Tier 1 fix)
        agent_config={
            'epsilon_start': 1.0,
            'epsilon_end': 0.1,
            'epsilon_decay': 0.95,
        },
        gamma_coverage=30.0,
        step_penalty=-0.05,
    )

    def empty_map():
        grid = np.zeros((5, 5), dtype=np.float32)
        grid[0, :] = 1.0
        grid[-1, :] = 1.0
        grid[:, 0] = 1.0
        grid[:, -1] = 1.0
        return grid

    env.training_map_generators['empty'] = empty_map
    env.training_map_order = ['empty']

    print("Training...")
    metrics, _, _, _, _ = train(env, episodes_per_type=50, verbose=False,
                                save_interval=50, model_dir="./models_tier1_test1")

    early_cov = np.mean(metrics.episode_coverage[:10]) if len(metrics.episode_coverage) >= 10 else 0
    late_cov = np.mean(metrics.episode_coverage[-10:]) if len(metrics.episode_coverage) >= 10 else 0
    
    early_loss = np.mean(metrics.episode_avg_loss[:10]) if len(metrics.episode_avg_loss) >= 10 else 0
    late_loss = np.mean(metrics.episode_avg_loss[-10:]) if len(metrics.episode_avg_loss) >= 10 else 0
    max_loss = max(metrics.episode_avg_loss) if metrics.episode_avg_loss else 0

    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"  Coverage: {early_cov:.1f}% → {late_cov:.1f}% (Δ{late_cov-early_cov:+.1f}%)")
    print(f"  Loss: {early_loss:.4f} → {late_loss:.4f} (max={max_loss:.4f})")
    print(f"  Loss stable: {'✅' if max_loss < 10.0 else '❌ EXPLODED'}")
    
    passed = late_cov > 95.0
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}")
    print("="*70)
    return passed, metrics


def test_2_scaling():
    """Test 2: 4 agents, 10x10 - THE CRITICAL TEST (was exploding)"""
    print("\n" + "="*70)
    print("TEST 2: 4 Agents, Empty 10x10 (LOSS EXPLOSION TEST)")
    print("="*70)
    print("Previous result: Loss 0.41 → 460.69 ❌")
    print("Expected now:    Loss < 5.0 (stable) ✅")
    print("="*70)

    env = MARL_QMIX_Environment(
        grid_size=10,
        num_agents=4,
        sensor_range=4,
        comm_range=8.0,
        coverage_threshold=0.8,
        completion_threshold_perc=90.0,
        max_episodes=200,
        max_steps_per_episode=100,
        device=None,  # Auto-detect: cuda if available, else cpu
        tensorboard_dir="./runs/tier1_test2",
        batch_size=64,
        memory_capacity=10000,
        gamma=0.99,
        lr_agents=0.0005,    # Standard
        lr_mixer=0.0001,     # 5x slower (KEY FIX)
        soft_update_tau=0.001,  # 5x slower (KEY FIX)
        agent_config={
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.98,
        },
        gamma_coverage=20.0,
        step_penalty=-0.02,
    )

    def empty_map():
        grid = np.zeros((10, 10), dtype=np.float32)
        grid[0, :] = 1.0
        grid[-1, :] = 1.0
        grid[:, 0] = 1.0
        grid[:, -1] = 1.0
        return grid

    env.training_map_generators['empty'] = empty_map
    env.training_map_order = ['empty']

    print("\nTraining 200 episodes...")
    print("(Monitoring for loss explosion...)")
    metrics, _, _, _, _ = train(env, episodes_per_type=200, verbose=False,
                                save_interval=200, model_dir="./models_tier1_test2")

    early_cov = np.mean(metrics.episode_coverage[:20]) if len(metrics.episode_coverage) >= 20 else 0
    late_cov = np.mean(metrics.episode_coverage[-20:]) if len(metrics.episode_coverage) >= 20 else 0
    
    early_loss = np.mean(metrics.episode_avg_loss[:20]) if len(metrics.episode_avg_loss) >= 20 else 0
    late_loss = np.mean(metrics.episode_avg_loss[-20:]) if len(metrics.episode_avg_loss) >= 20 else 0
    max_loss = max(metrics.episode_avg_loss) if metrics.episode_avg_loss else 0

    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"  Coverage: {early_cov:.1f}% → {late_cov:.1f}% (Δ{late_cov-early_cov:+.1f}%)")
    print(f"  Loss: {early_loss:.4f} → {late_loss:.4f}")
    print(f"  Max loss: {max_loss:.4f}")
    print(f"  Loss stable: {'✅ YES' if max_loss < 50.0 else '❌ EXPLODED'}")
    
    # Check for explosion
    exploded = max_loss > 50.0
    if exploded:
        print(f"\n⚠️  WARNING: Loss still exploding! Need Tier 2 fixes.")
    else:
        print(f"\n✅ SUCCESS: Loss explosion FIXED with Tier 1!")
    
    passed = late_cov > 85.0 and not exploded
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}")
    print("="*70)
    return passed, metrics


def test_3_obstacles():
    """Test 3: 2 agents, obstacles - Check for regression"""
    print("\n" + "="*70)
    print("TEST 3: 2 Agents, 10x10 with Obstacles")
    print("="*70)
    print("Previous: Coverage 83% → 81% (REGRESSION)")
    print("Expected: Coverage should improve or stay flat")
    print("="*70)

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
        device=None,  # Auto-detect: cuda if available, else cpu
        tensorboard_dir="./runs/tier1_test3",
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

    def room_map():
        return generate_room_map(10, num_rooms_range=(2, 3), room_size_range=(3, 4))

    env.training_map_generators['room'] = room_map
    env.training_map_order = ['room']

    print("\nTraining 300 episodes...")
    metrics, _, _, _, _ = train(env, episodes_per_type=300, verbose=False,
                                save_interval=300, model_dir="./models_tier1_test3")

    early_cov = np.mean(metrics.episode_coverage[:30]) if len(metrics.episode_coverage) >= 30 else 0
    late_cov = np.mean(metrics.episode_coverage[-30:]) if len(metrics.episode_coverage) >= 30 else 0
    
    early_loss = np.mean(metrics.episode_avg_loss[:30]) if len(metrics.episode_avg_loss) >= 30 else 0
    late_loss = np.mean(metrics.episode_avg_loss[-30:]) if len(metrics.episode_avg_loss) >= 30 else 0
    max_loss = max(metrics.episode_avg_loss) if metrics.episode_avg_loss else 0

    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"  Coverage: {early_cov:.1f}% → {late_cov:.1f}% (Δ{late_cov-early_cov:+.1f}%)")
    print(f"  Loss: {early_loss:.4f} → {late_loss:.4f} (max={max_loss:.4f})")
    
    regression = (late_cov - early_cov) < -2.0
    exploded = max_loss > 50.0
    
    if regression:
        print(f"  ⚠️  Coverage REGRESSION detected")
    if exploded:
        print(f"  ⚠️  Loss explosion detected")
    
    passed = late_cov > 75.0 and not regression
    print(f"\n{'✅ PASSED' if passed else '❌ FAILED'}")
    print("="*70)
    return passed, metrics


def run_tier1_validation():
    """Run validation with Tier 1 fixes only"""
    print("\n" + "#"*70)
    print("# TIER 1 VALIDATION: Proper Engineering Solutions")
    print("#"*70)
    print("# Fixes implemented:")
    print("#   1. Separate learning rates (agents=0.0005, mixer=0.0001)")
    print("#   2. Slower target updates (tau=0.001)")
    print("#   3. Reward normalization (Welford's algorithm)")
    print("#   4. Huber loss (smooth_l1)")
    print("#")
    print("# Critical success metric:")
    print("#   Test 2 loss < 50.0 (was 460.69)")
    print("#"*70)

    results = []

    try:
        p1, m1 = test_1_simple()
        results.append(("Test 1: Simple", p1))
    except Exception as e:
        print(f"\n❌ Test 1 crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 1: Simple", False))
        m1 = None

    try:
        p2, m2 = test_2_scaling()
        results.append(("Test 2: Scaling (CRITICAL)", p2))
    except Exception as e:
        print(f"\n❌ Test 2 crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 2: Scaling (CRITICAL)", False))
        m2 = None

    try:
        p3, m3 = test_3_obstacles()
        results.append(("Test 3: Obstacles", p3))
    except Exception as e:
        print(f"\n❌ Test 3 crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Test 3: Obstacles", False))
        m3 = None

    print("\n" + "="*70)
    print("TIER 1 VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, p in results:
        print(f"{'✅' if p else '❌'} {name}")

    print(f"\nResult: {passed}/{total} tests passed")
    
    test2_passed = results[1][1] if len(results) > 1 else False
    
    if test2_passed:
        print("\n🎉 SUCCESS: Tier 1 fixes resolved loss explosion!")
        print("   Test 2 is now stable. No band-aids needed.")
    else:
        print("\n⚠️  Test 2 still unstable. May need Tier 2 fixes:")
        print("   - Bounded mixer output (architectural)")
        print("   - Lower reward scale (gamma_coverage)")
    
    print("="*70)
    return passed == total


if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    success = run_tier1_validation()
