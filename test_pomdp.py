"""
POMDP Integrity Validation Tests

Tests to verify that agents have true partial observability:
1. Agent observations match local maps
2. No omniscient obstacle knowledge
3. Global state uses only shared knowledge
4. Raycasting discovers obstacles
"""

import numpy as np
import torch
from environment import MARL_QMIX_Environment
from config import GRID_SIZE, NUM_AGENTS, SENSOR_RANGE


def test_local_map_consistency():
    """Test 1: Verify agent's network input matches its local_map."""
    print("\n🧪 Test 1: Local Map Consistency")
    print("=" * 60)

    env = MARL_QMIX_Environment(
        grid_size=20, num_agents=4, sensor_range=5,
        device="cpu", tensorboard_dir=None
    )
    env.reset(full_reset=True)

    # Run a few steps
    for _ in range(10):
        actions = {i: (1, 0) for i in range(4)}
        env.step(actions)

    # Check each agent
    all_passed = True
    for agent in env.agents:
        grid_tensor, _ = agent.get_state_tensor()
        coverage_grid = grid_tensor[0].cpu().numpy()
        obstacle_grid = grid_tensor[1].cpu().numpy()

        # Build expected grids from local_map
        expected_coverage = np.zeros_like(coverage_grid)
        expected_obstacles = np.zeros_like(obstacle_grid)

        for node, data in agent.local_map.nodes(data=True):
            r, c = node
            if 0 <= r < env.grid_size and 0 <= c < env.grid_size:
                expected_coverage[r, c] = data.get('pc', 0.0)
                if data.get('type') == 'occupied':
                    expected_obstacles[r, c] = 1.0

        # Check matches
        coverage_match = np.allclose(coverage_grid, expected_coverage)
        obstacle_match = np.allclose(obstacle_grid, expected_obstacles)

        if not coverage_match or not obstacle_match:
            print(f"❌ Agent {agent.agent_id} FAILED:")
            if not coverage_match:
                print(f"   Coverage mismatch: {np.sum(np.abs(coverage_grid - expected_coverage))} errors")
            if not obstacle_match:
                print(f"   Obstacle mismatch: {np.sum(np.abs(obstacle_grid - expected_obstacles))} errors")
            all_passed = False
        else:
            print(f"✅ Agent {agent.agent_id} PASSED")

    if all_passed:
        print("\n✅ Test 1 PASSED: All agent observations match local maps")
    else:
        print("\n❌ Test 1 FAILED: Observation mismatch detected")

    return all_passed


def test_no_obstacle_omniscience():
    """Test 2: Agent should NOT know about obstacles it hasn't observed."""
    print("\n🧪 Test 2: No Omniscient Obstacle Knowledge")
    print("=" * 60)

    from utils import generate_room_map

    env = MARL_QMIX_Environment(
        grid_size=20, num_agents=1, sensor_range=5,
        device="cpu", tensorboard_dir=None
    )

    # Generate map with obstacles
    env.obstacle_grid = generate_room_map(20)
    env.world_state = env._initialize_world_state()
    env.agent_positions = {0: (1, 1)}

    # Create agent
    env.agents = env._create_agents()
    env.agents[0]._initialize_local_map()

    # Get agent's obstacle knowledge BEFORE exploration
    grid_tensor, _ = env.agents[0].get_state_tensor()
    obstacle_grid_before = grid_tensor[1].cpu().numpy()

    # Count obstacles in agent's initial local map
    known_obstacles_before = np.sum(obstacle_grid_before == 1.0)

    # Count obstacles in ground truth
    total_obstacles = np.sum(env.obstacle_grid == 1.0)

    print(f"Ground truth obstacles: {total_obstacles}")
    print(f"Agent initial knowledge: {known_obstacles_before} obstacles")

    # Agent should know VERY FEW obstacles initially (maybe none or just nearby)
    if known_obstacles_before < total_obstacles * 0.3:  # Less than 30% knowledge
        print(f"✅ Test 2 PASSED: Agent knows only {known_obstacles_before}/{total_obstacles} obstacles initially ({known_obstacles_before/max(total_obstacles,1)*100:.1f}%)")
        return True
    else:
        print(f"❌ Test 2 FAILED: Agent knows {known_obstacles_before}/{total_obstacles} obstacles ({known_obstacles_before/max(total_obstacles,1)*100:.1f}%) - too much!")
        return False


def test_global_state_integrity():
    """Test 3: Verify global state contains only shared (observed) information."""
    print("\n🧪 Test 3: Global State Integrity (Shared Knowledge Only)")
    print("=" * 60)

    env = MARL_QMIX_Environment(
        grid_size=20, num_agents=4, sensor_range=5,
        device="cpu", tensorboard_dir=None
    )
    states = env.reset(full_reset=True)

    # Run steps
    for _ in range(20):
        actions = {i: (1, 0) for i in range(4)}
        env.step(actions)

    # Count obstacles in shared knowledge
    shared_obstacles = set()
    for agent in env.agents:
        for node, data in agent.local_map.nodes(data=True):
            if data.get('type') == 'occupied':
                shared_obstacles.add(node)

    # Count obstacles in ground truth
    ground_truth_obstacles = set()
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            if env.obstacle_grid[r, c] == 1.0:
                ground_truth_obstacles.add((r, c))

    # Shared knowledge should be SUBSET of ground truth
    is_subset = shared_obstacles.issubset(ground_truth_obstacles)

    # Should NOT be complete knowledge
    coverage_ratio = len(shared_obstacles) / max(len(ground_truth_obstacles), 1)

    print(f"Ground truth obstacles: {len(ground_truth_obstacles)}")
    print(f"Shared knowledge obstacles: {len(shared_obstacles)}")
    print(f"Coverage ratio: {coverage_ratio*100:.1f}%")

    if is_subset and coverage_ratio < 0.95:  # Less than 95% knowledge
        print(f"✅ Test 3 PASSED: Global state has partial knowledge ({coverage_ratio*100:.1f}%)")
        return True
    elif not is_subset:
        print(f"❌ Test 3 FAILED: Global state contains obstacles not in ground truth!")
        return False
    else:
        print(f"❌ Test 3 FAILED: Global state has too much knowledge ({coverage_ratio*100:.1f}%)")
        return False


def test_raycasting_discovers_obstacles():
    """Test 4: Verify raycasting updates local maps with discovered obstacles."""
    print("\n🧪 Test 4: Raycasting Discovers Obstacles")
    print("=" * 60)

    from data_structures import RobotState

    env = MARL_QMIX_Environment(
        grid_size=20, num_agents=1, sensor_range=5,
        device="cpu", tensorboard_dir=None
    )
    env.reset(full_reset=True)

    agent = env.agents[0]

    # Count obstacles before exploration
    obstacles_before = sum(
        1 for _, data in agent.local_map.nodes(data=True)
        if data.get('type') == 'occupied'
    )

    print(f"Obstacles known before exploration: {obstacles_before}")

    # Move agent to explore
    for step in range(30):
        valid_actions = env.get_valid_actions(0)
        action = valid_actions[0] if valid_actions else (0, 0)
        env.step({0: action})

    # Count obstacles after exploration
    obstacles_after = sum(
        1 for _, data in agent.local_map.nodes(data=True)
        if data.get('type') == 'occupied'
    )

    print(f"Obstacles known after exploration: {obstacles_after}")
    print(f"Obstacles discovered: {obstacles_after - obstacles_before}")

    # Agent should have discovered some obstacles
    if obstacles_after > obstacles_before:
        print(f"✅ Test 4 PASSED: Agent discovered {obstacles_after - obstacles_before} obstacles")
        return True
    else:
        print(f"❌ Test 4 FAILED: Agent did not discover obstacles ({obstacles_before} → {obstacles_after})")
        return False


def run_all_tests():
    """Run all POMDP integrity tests."""
    print("\n" + "=" * 60)
    print("POMDP INTEGRITY VALIDATION TESTS")
    print("=" * 60)

    results = []

    results.append(("Local Map Consistency", test_local_map_consistency()))
    results.append(("No Omniscient Knowledge", test_no_obstacle_omniscience()))
    results.append(("Global State Integrity", test_global_state_integrity()))
    results.append(("Raycasting Discovers Obstacles", test_raycasting_discovers_obstacles()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! POMDP integrity verified!")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. POMDP violations detected!")
        return False


if __name__ == "__main__":
    run_all_tests()
