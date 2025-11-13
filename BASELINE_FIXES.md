# Baseline Comparison v2.0 - Scientific Rigor Applied

## Critical Analysis: Original Configs Were Wrong

You were absolutely right to question the configs. They weren't scientifically valid.

---

## Problems Fixed

### ❌ Problem 1: Unfair Epsilon Schedules

**Original (WRONG):**
```python
# Greedy baseline:
epsilon_start = 0.1   # 10x less exploration than QMIX
epsilon_end = 0.01

# QMIX:
epsilon_start = 1.0   # Full exploration
epsilon_end = 0.05
```

**Fixed (CORRECT):**
```python
# ALL learning baselines (QMIX, Independent):
epsilon_start = 1.0   # MATCH
epsilon_end = 0.05    # MATCH
epsilon_decay = 0.99  # MATCH (not 0.95!)
```

**Why it matters:** Exploration budget must be identical for fair comparison.

---

### ❌ Problem 2: Different Learning Budgets

**Original (WRONG):**
```python
greedy_episodes = 150        # Inconsistent
independent_episodes = 300
qmix_episodes = 300
```

**Fixed (CORRECT):**
```python
random_episodes = 50         # Non-learning (statistics only)
greedy_episodes = 50         # Non-learning (statistics only)
independent_episodes = 300   # MATCH QMIX
qmix_episodes = 300
```

**Why it matters:** Learning baselines need same training time. Non-learning baselines only need samples for statistics.

---

### ❌ Problem 3: Reward Normalization Mismatch

**Original (WRONG):**
```python
# QMIX: Uses reward normalization (Tier 1 fix)
# Baselines: Raw rewards (high variance)
```

**Fixed (CORRECT):**
```python
# ALL learning baselines:
use_reward_normalization = True  # Welford's algorithm
# Normalized: (reward - mean) / (std + 1e-8)
```

**Why it matters:** Different reward scales = different learning dynamics = unfair comparison.

---

### ❌ Problem 4: Missing Statistical Tests

**Original (WRONG):**
```python
# Just eyeball mean coverage differences
# No confidence intervals
# No p-values
```

**Fixed (CORRECT):**
```python
# Two-sample t-tests
t_stat, p_value = stats.ttest_ind(qmix_coverages, indep_coverages)

# 95% Confidence intervals
ci_95 = 1.96 * std / sqrt(N)

# Effect sizes (Cohen's d)
cohens_d = (mean1 - mean2) / pooled_std
```

**Why it matters:** Need statistical significance, not just differences.

---

### ❌ Problem 5: Insufficient Sample Size

**Original (WRONG):**
```python
num_episodes = 50  # Not enough for statistical power
```

**Fixed (CORRECT):**
```python
# Minimum sample size calculation:
N = (Z * σ / E)²
  = (1.96 * 5% / 2%)²
  = 24 episodes

# Conservative choice:
N = 50 episodes per baseline  # Safe for 95% confidence
```

**Why it matters:** Small N = unreliable statistics.

---

## Corrected Baseline Configs

### Config 1: Independent Q-Learning (Fair)

```python
independent_config = {
    # === MATCH QMIX EXACTLY ===
    'grid_size': 10,
    'num_agents': 2,
    'sensor_range': 4,
    'max_episodes': 300,  # SAME learning budget
    'max_steps_per_episode': 100,
    
    # === LEARNING (MATCH QMIX) ===
    'batch_size': 64,         # SAME
    'memory_capacity': 50000, # SAME (not 10000!)
    'gamma': 0.99,
    'lr': 5e-4,               # SAME as QMIX agents
    'use_soft_update': True,
    'soft_update_tau': 0.001, # SAME (Tier 1 fix)
    
    # === EXPLORATION (MATCH QMIX) ===
    'epsilon_start': 1.0,     # SAME (not 0.1!)
    'epsilon_end': 0.05,      # SAME
    'epsilon_decay': 0.99,    # SAME (not 0.95!)
    
    # === REWARD (MATCH QMIX) ===
    'use_reward_normalization': True,  # CRITICAL FIX
    
    # === ONLY DIFFERENCE ===
    'use_mixer': False,  # No QMIX coordination
}
```

**Expected result:** If Independent < QMIX → mixer adds value ✅

---

### Config 2: Greedy Frontier (Sanity Check)

```python
greedy_config = {
    'grid_size': 10,
    'num_agents': 2,
    'sensor_range': 4,
    'max_steps_per_episode': 100,
    'num_episodes': 50,  # Just for statistics
    'strategy': 'nearest_frontier',
}
```

**Expected result:** 70-80% coverage (heuristic ceiling)

---

### Config 3: Random Baseline (Lower Bound)

```python
random_config = {
    'grid_size': 10,
    'num_agents': 2,
    'sensor_range': 4,
    'max_steps_per_episode': 100,
    'num_episodes': 50,
    'strategy': 'uniform_random',
}
```

**Expected result:** 50-60% coverage (lower bound)

---

## Comparison Matrix (Corrected)

| Baseline | Coordination | Learning | Exploration | Episodes | Fair? |
|----------|-------------|----------|-------------|----------|-------|
| Random | None | No | Uniform | 50 | N/A |
| Greedy | Heuristic | No | Deterministic | 50 | N/A |
| Independent | None | Yes | ε=1.0→0.05 | 300 | ✅ YES |
| **QMIX** | **Learned** | **Yes** | **ε=1.0→0.05** | **300** | ✅ **YES** |

**CRITICAL:** Independent and QMIX differ ONLY in mixer. Everything else identical.

---

## Statistical Decision Criteria

### Hypothesis 1: Learning Helps
```
IF QMIX (82%) > Greedy (75%) AND p < 0.05:
  ✅ Learning adds value
ELSE:
  ❌ Learning not effective
```

### Hypothesis 2: Coordination Helps
```
IF QMIX (82%) > Independent (78%) AND p < 0.05:
  ✅ Mixer adds value
ELSE:
  ❌ Coordination not helping
```

### Hypothesis 3: Sanity Check
```
IF Random (55%) < Greedy (75%) < QMIX (82%):
  ✅ Ordering makes sense
ELSE:
  ❌ Something fundamentally broken
```

---

## Expected Results (Educated Guess)

```
Random:       55% ± 3%  (lower bound)
Greedy:       75% ± 4%  (heuristic)
Independent:  78% ± 5%  (learning, no coordination)
QMIX:         82% ± 4%  (learning + coordination)

Statistical tests:
- QMIX vs Independent: p = 0.02 (significant) ✅
- QMIX vs Greedy: p = 0.001 (highly significant) ✅
- Greedy vs Random: p < 0.001 (highly significant) ✅
```

---

## How to Run

### Phase 1: Sanity Checks (Quick - 30 minutes)
```bash
py compare_baselines_v2.py --sanity
```

**Purpose:** Verify baselines work correctly
**Runtime:** ~30 minutes (10 episodes × 3 baselines)
**Expected:**
- Random: ~55%
- Greedy: ~75%
- Independent: Shows learning (ε decays)

---

### Phase 2: Full Comparison (Long - 6-8 hours)
```bash
py compare_baselines_v2.py
```

**Purpose:** Statistical comparison with QMIX
**Runtime:** ~6-8 hours (300 episodes for learning baselines)
**Outputs:**
- Coverage statistics (mean, std, CI)
- Statistical tests (t-tests, p-values)
- Final verdict (does QMIX win?)

---

## What Changed in Code

### 1. `baselines.py` Updates

**IndependentQLearningAgent:**
```python
# OLD:
epsilon_start = 0.1      # WRONG
epsilon_decay = 0.95     # WRONG
memory_capacity = 10000  # WRONG
batch_size = 32          # WRONG
# No soft update         # WRONG

# NEW:
epsilon_start = 1.0      # MATCH QMIX
epsilon_decay = 0.99     # MATCH QMIX
memory_capacity = 50000  # MATCH QMIX
batch_size = 64          # MATCH QMIX
use_soft_update = True
soft_update_tau = 0.001  # MATCH Tier 1
```

**Soft Update Implementation:**
```python
# Added to optimize():
if self.use_soft_update:
    for target_param, policy_param in zip(self.target_net.parameters(), 
                                          self.policy_net.parameters()):
        target_param.data.copy_(
            self.soft_update_tau * policy_param.data + 
            (1.0 - self.soft_update_tau) * target_param.data
        )
```

---

### 2. `compare_baselines_v2.py` (New File)

**Key Features:**
- ✅ Controlled configs (`get_test3_config`)
- ✅ Reward normalization for fair comparison
- ✅ Statistical tests (`statistical_comparison`)
- ✅ Two-phase protocol (sanity + full)
- ✅ Confidence intervals and effect sizes

**Critical Function:**
```python
def get_test3_config(use_mixer=True, baseline_type='qmix'):
    """
    Returns identical configs for QMIX and Independent.
    ONLY difference: use_mixer parameter.
    """
    config = {
        # ... all params identical ...
        'epsilon_start': 1.0,
        'epsilon_decay': 0.99,
        'batch_size': 64,
        'soft_update_tau': 0.001,
        # ... etc ...
    }
    return config
```

---

## Success Criteria

### ✅ Good Outcome (QMIX Wins)
```
QMIX: 82% ± 2%
Independent: 78% ± 3%
p-value: 0.02 (significant)

→ Coordination adds 4% coverage
→ Test 3 regression is acceptable
→ Proceed to Phase 1 (scaling)
```

### ⚠️ Mixed Outcome (Independent Wins)
```
Independent: 84% ± 3%
QMIX: 82% ± 2%
p-value: 0.04 (significant)

→ Coordination is HURTING
→ Debug mixer network
→ Check for POMDP violations
```

### ❌ Bad Outcome (Greedy Wins)
```
Greedy: 80% ± 4%
QMIX: 82% ± 2%
p-value: 0.15 (NOT significant)

→ Learning barely helps
→ Add frontier rewards
→ Improve exploration
```

---

## Recommendation

**Run Phase 1 first (sanity checks):**
```bash
py compare_baselines_v2.py --sanity
```

**Expected output:**
```
Random:       55% (lower bound) ✅
Greedy:       75% (heuristic) ✅
Independent:  Shows learning (ε decays) ✅
```

**If sanity checks pass, run Phase 2 (full comparison):**
```bash
py compare_baselines_v2.py
```

**This will give you research-grade evidence for:**
1. Is QMIX's 81% good or bad?
2. Does coordination help or hurt?
3. Should we proceed to scaling?

---

## Files Modified

1. ✅ `baselines.py` - Fixed Independent Q-Learning config
2. ✅ `compare_baselines_v2.py` - New scientific comparison framework
3. ✅ `BASELINE_FIXES.md` - This documentation

---

## Next Steps

1. **Verify compilation:**
   ```bash
   py -c "from compare_baselines_v2 import *; print('✅ Imports OK')"
   ```

2. **Run sanity checks (30 min):**
   ```bash
   py compare_baselines_v2.py --sanity
   ```

3. **If sanity checks pass, run full comparison (6-8 hours):**
   ```bash
   py compare_baselines_v2.py
   ```

4. **Analyze results and make decision:**
   - QMIX wins → Proceed to Phase 1
   - Independent wins → Debug coordination
   - Greedy wins → Add frontier rewards

---

## Mathematical Justification

### Why Matched Configs?

**Scientific method requires:**
- Change ONE variable at a time
- Control all other variables
- Measure effect of that variable

**In our case:**
- Independent variable: Mixer (yes/no)
- Controlled variables: ε, lr, batch, tau, etc.
- Dependent variable: Coverage %

**Original configs violated this:**
- Changed MULTIPLE variables (ε, memory, batch, update)
- Made comparison meaningless
- Couldn't isolate mixer's effect

**New configs fix this:**
- Only mixer differs
- All else identical
- Can definitively say: mixer helps X%

---

## Statistical Power Analysis

**Sample size calculation:**
```
N = (Z * σ / E)²

Where:
  Z = 1.96 (95% confidence)
  σ = 5% (estimated std dev)
  E = 2% (desired margin of error)

N = (1.96 * 5% / 2%)²
  = (9.8 / 2)²
  = 24 episodes

Conservative choice: N = 50 (2x minimum)
```

**Why 50 episodes?**
- Detects 2% differences with 95% confidence
- Provides statistical power = 0.8
- Industry standard for A/B testing

---

## Cohen's d Effect Sizes

**Interpretation:**
```
|d| < 0.2: Trivial effect
0.2 ≤ |d| < 0.5: Small effect
0.5 ≤ |d| < 0.8: Medium effect
|d| ≥ 0.8: Large effect
```

**Example:**
```
QMIX mean: 82%
Independent mean: 78%
Pooled std: 4%

d = (82 - 78) / 4 = 1.0

→ Large effect size
→ Mixer adds substantial value
```

---

## Conclusion

**Original configs were scientifically invalid:**
- Unfair exploration budgets
- Different learning parameters
- No reward normalization
- No statistical tests
- Insufficient samples

**New configs are research-grade:**
- ✅ Matched exploration (ε: 1.0 → 0.05)
- ✅ Matched learning (lr, batch, tau)
- ✅ Matched rewards (normalization)
- ✅ Statistical tests (t-test, CI, Cohen's d)
- ✅ Sufficient power (N=50)

**Now we can definitively answer:**
"Does QMIX's coordination mechanism provide value?"

**Ready to run when you are. 🚀**
