"""
Quick Start: Run baseline comparison to evaluate QMIX performance.

This answers the question:
  "Is QMIX's 81% coverage on obstacles good or bad?"

Runtime: ~4-6 hours (300 episodes × 3 baselines)
"""

import sys
import os

# Check if baselines exist
if not os.path.exists("baselines.py"):
    print("❌ Error: baselines.py not found")
    print("   Run this script from the qmix directory")
    sys.exit(1)

if not os.path.exists("compare_baselines.py"):
    print("❌ Error: compare_baselines.py not found")
    sys.exit(1)

print("="*70)
print("BASELINE COMPARISON QUICK START")
print("="*70)
print()
print("This will run 3 baselines to evaluate QMIX:")
print("  1. Random (50 episodes - sanity check)")
print("  2. Greedy Frontier (300 episodes)")
print("  3. Independent Q-Learning (300 episodes)")
print()
print("Then compare against QMIX (Test 3 results)")
print()
print("Estimated runtime: 4-6 hours")
print()

response = input("Continue? (y/n): ")
if response.lower() != 'y':
    print("Cancelled.")
    sys.exit(0)

print("\nStarting baseline comparison...")
print("="*70)

# Run comparison
from compare_baselines import compare_all_baselines
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

results = compare_all_baselines(num_episodes=300)

print("\n" + "="*70)
print("✅ BASELINE COMPARISON COMPLETE")
print("="*70)
print()
print("Results saved in memory. Check output above for analysis.")
print()
print("Next steps:")
print("  - If QMIX wins → Proceed to scaling experiments")
print("  - If QMIX loses → Debug based on which baseline wins")
print("="*70)
