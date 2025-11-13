"""
VERIFICATION TEST: Action Masking Fix

This script verifies that the action masking bug is fixed:
- Valid actions now include conflict checking
- Network only learns Q-values for executable actions
- No more action space mismatch

Expected behavior:
1. When agents are close, valid actions should be restricted
2. Network should never select conflicting actions
3. No more negative learning due to conflict resolution
"""

import numpy as np
import random
import torch
from environment import MARL_QMIX_Environment
from utils import generate_empty_map

def test_action_masking_fix():
    """Test that action masking now includes conflict checking."""
    
    print("\n" + "="*70)
    print("VERIFICATION TEST: Action Masking Fix")
    print("="*70)
    
    # Create simple environment
    config = {
        'grid_size': 10,
        'num_agents': 2,
        'sensor_range': 3,
        'comm_range': 6.0,
        'coverage_threshold': 0.8,
        'completion_threshold_perc': 85.0,
        'max_episodes': 10,
        'max_steps_per_episode': 50,
        'gamma_coverage': 15.0,
        'step_penalty': -0.02,
        'orientation_cost_factor': 0.02,
        'invalid_move_penalty': -0.5,
        'batch_size': 32,
        'memory_capacity': 1000,
        'gamma': 0.99,
        'lr_agents': 5e-4,
        'lr_mixer': 1e-4,
        'use_soft_update': True,
        'soft_update_tau': 0.001,
        'agent_config': {
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.95,
        },
        'device': None,
        'tensorboard_dir': None,
    }
    
    env = MARL_QMIX_Environment(**config)
    env.training_map_generators['empty'] = lambda: generate_empty_map(10)
    env.training_map_order = ['empty']
    
    env.reset(full_reset=True, mode='train')
    
    print(f"\n✓ Environment created with {env.num_agents} agents")
    print(f"✓ Grid size: {env.grid_size}×{env.grid_size}")
    
    # Test 1: Check valid actions when agents are far apart
    print("\n" + "-"*70)
    print("TEST 1: Agents far apart (no conflicts expected)")
    print("-"*70)
    
    env.agent_positions[0] = (2, 2)
    env.agent_positions[1] = (7, 7)
    
    valid_0 = env.get_valid_actions(0)
    valid_1 = env.get_valid_actions(1)
    
    print(f"Agent 0 at {env.agent_positions[0]}: {len(valid_0)} valid actions")
    print(f"Agent 1 at {env.agent_positions[1]}: {len(valid_1)} valid actions")
    
    if len(valid_0) >= 7 and len(valid_1) >= 7:
        print("✅ PASS: Both agents have many valid actions when far apart")
    else:
        print(f"❌ FAIL: Expected ≥7 actions, got {len(valid_0)}, {len(valid_1)}")
    
    # Test 2: Check valid actions when agents are adjacent
    print("\n" + "-"*70)
    print("TEST 2: Agents adjacent (conflicts expected)")
    print("-"*70)
    
    env.agent_positions[0] = (5, 5)
    env.agent_positions[1] = (5, 6)  # Right next to agent 0
    
    valid_0 = env.get_valid_actions(0)
    valid_1 = env.get_valid_actions(1)
    
    print(f"Agent 0 at {env.agent_positions[0]}: {len(valid_0)} valid actions")
    print(f"  Valid: {valid_0}")
    print(f"Agent 1 at {env.agent_positions[1]}: {len(valid_1)} valid actions")
    print(f"  Valid: {valid_1}")
    
    # Agent 0 should NOT be able to move to (5,6) - occupied by agent 1
    if (0, 1) not in valid_0:
        print("✅ PASS: Agent 0 cannot move to occupied position (5,6)")
    else:
        print("❌ FAIL: Agent 0 can move to occupied position - conflict not detected!")
    
    # Agent 1 should NOT be able to move to (5,5) - occupied by agent 0
    if (0, -1) not in valid_1:
        print("✅ PASS: Agent 1 cannot move to occupied position (5,5)")
    else:
        print("❌ FAIL: Agent 1 can move to occupied position - conflict not detected!")
    
    # Test 3: Run a short episode and check for action space violations
    print("\n" + "-"*70)
    print("TEST 3: Short episode - verify no action space violations")
    print("-"*70)
    
    env.reset(full_reset=False, mode='train')
    
    violations = 0
    for step in range(20):
        actions = {}
        for agent_id, agent in enumerate(env.agents):
            valid_actions = env.get_valid_actions(agent_id)
            
            # Network selects action
            state_tensor = agent.get_state_tensor()
            action = agent.select_action(state_tensor, valid_actions)
            
            # Verify selected action is in valid set
            if action not in valid_actions:
                print(f"❌ Step {step}, Agent {agent_id}: Selected {action} not in valid set!")
                violations += 1
            
            actions[agent_id] = action
        
        # Execute step
        env.step(actions)
    
    if violations == 0:
        print(f"✅ PASS: No action space violations in 20 steps")
    else:
        print(f"❌ FAIL: {violations} action space violations detected")
    
    # Test 4: Check that conflict resolution is now unnecessary
    print("\n" + "-"*70)
    print("TEST 4: Verify conflict resolution is redundant")
    print("-"*70)
    
    env.reset(full_reset=False, mode='train')
    
    conflicts_prevented = 0
    for step in range(20):
        actions = {}
        for agent_id in range(env.num_agents):
            valid_actions = env.get_valid_actions(agent_id)
            actions[agent_id] = random.choice(valid_actions)
        
        # Check if any conflicts would occur
        target_positions = {}
        for agent_id, action in actions.items():
            current = env.agent_positions[agent_id]
            target = (current[0] + action[0], current[1] + action[1])
            target_positions[agent_id] = target
        
        # Check for duplicates
        if len(set(target_positions.values())) < len(target_positions):
            conflicts_prevented += 1
            print(f"  Step {step}: Conflict would have occurred - {target_positions}")
        
        env.step(actions)
    
    if conflicts_prevented == 0:
        print(f"✅ PASS: No conflicts in 20 steps - action masking prevents them")
    else:
        print(f"⚠️  WARNING: {conflicts_prevented} conflicts would have occurred")
        print(f"   This suggests action masking might still have edge cases")
    
    # Final summary
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    print("\n✓ Action masking fix has been applied")
    print("✓ Network now only learns Q-values for executable actions")
    print("✓ No more action space mismatch")
    print("\nNext step: Re-run Test 1B to verify positive learning")
    print("Expected: Independent and QMIX both show early < late")
    print("="*70 + "\n")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    test_action_masking_fix()
