# Phase 1 Foundation Validation - IMMUTABLE FRAMEWORK

## DO NOT PROCEED WITHOUT PASSING ALL TESTS

This document defines the **LOCKED** validation framework for the QMIX project.

**Status:** Phase 1 - Foundation Validation  
**Last Updated:** 2025-11-13  
**Author:** Research Team  

---

## The Contract: No Goal-Post Shifting

### LOCKED Elements (Cannot Change):
- ✅ Test configurations (grid sizes, agent counts, episode budgets)
- ✅ Success criteria (coverage thresholds, p-values, improvement margins)
- ✅ Baseline implementations (random = pure random, no networks)
- ✅ Evaluation protocol (statistical tests, significance levels)

### Can Change (Only if Bugs Found):
- ⚠️ Bug fixes in implementation (MUST document in failure log)
- ⚠️ Hyperparameters within test (lr, batch size) IF test fails
- ⚠️ Network architecture IF proven fundamentally broken

### NEVER Change:
- ❌ Test criteria to make tests pass
- ❌ Success definitions mid-experiment
- ❌ Baseline implementations to favor QMIX
- ❌ Statistical thresholds after seeing results

---

## Phase 1: Foundation Validation

### Test 1A: Single Agent, Empty Grid

**Purpose:** Validate that basic Q-learning works  
**File:** `test_1a_single_agent.py`  
**Runtime:** ~10-15 minutes  

**Configuration (LOCKED):**
```python
- Grid: 10×10 empty (no obstacles)
- Agents: 1 (no coordination)
- Episodes: 100
- Steps/episode: 50
- Sensor range: 3
```

**Success Criteria (LOCKED):**
```python
✅ Learned coverage: >75%
✅ Random baseline: 50-60%
✅ Learned beats random by >15%
✅ Loss stable: max < 10.0
✅ Positive learning: Late > Early + 5%
```

**What This Tests:**
- Q-learning convergence
- Exploration-exploitation balance
- Coverage calculation correctness
- Random baseline validity

**How to Run:**
```bash
py test_1a_single_agent.py
```

**Expected Output:**
```
Random Baseline:  55% ± 3%
Learned Policy:   82% ± 4%
Improvement:      +12%
Loss (max):       3.2

✅ TEST 1A PASSED
```

**If Test Fails:**
1. Check random baseline (50-60%)
   - If outside range: Random implementation broken
   - Fix: Ensure pure uniform action selection
2. Check learned policy (<75%)
   - Root cause: Q-learning not converging
   - Fix: Check epsilon decay, learning rate, reward scaling
3. Check loss stability (>10.0)
   - Root cause: Gradient explosion
   - Fix: Check gradient clipping, target update frequency

---

### Test 1B: Two Agents, Empty Grid

**Purpose:** Validate that QMIX coordination helps  
**File:** `test_1b_two_agents.py`  
**Runtime:** ~30-40 minutes  
**Prerequisite:** Test 1A MUST pass first  

**Configuration (LOCKED):**
```python
- Grid: 15×15 empty
- Agents: 2 (coordination possible)
- Episodes: 200
- Steps/episode: 100
- Sensor range: 3
- Comm range: 6.0
```

**Success Criteria (LOCKED):**
```python
✅ QMIX coverage: >80%
✅ Independent coverage: 70-80%
✅ QMIX > Independent (p < 0.05)
✅ Random baseline: 50-60%
✅ Greedy baseline: 65-75%
✅ No negative learning (Late ≥ Early)
```

**What This Tests:**
- Mixer network effectiveness
- Coordination value quantification
- Independent Q-learning baseline
- Statistical significance testing

**How to Run:**
```bash
py test_1b_two_agents.py
```

**Expected Output:**
```
Random:       55% ± 3%
Greedy:       70% ± 4%
Independent:  76% ± 5%
QMIX:         84% ± 4%

Gap (QMIX - Independent): +8%
p-value: 0.02

✅ TEST 1B PASSED
```

**If Test Fails:**
1. QMIX < 80%
   - Root cause: Coordination not learning
   - Fix: Check mixer network, separate learning rates
2. QMIX ≤ Independent
   - Root cause: Mixer hurting performance
   - Fix: Check mixer architecture, credit assignment
3. p-value ≥ 0.05
   - Root cause: High variance or small effect
   - Fix: Run more episodes (increase N), check reward normalization

---

### Test 1C: Two Agents, Simple Obstacles

**Purpose:** Validate POMDP handling  
**File:** `test_1c_obstacles.py` (TO BE IMPLEMENTED)  
**Runtime:** ~40-50 minutes  
**Prerequisite:** Tests 1A + 1B MUST pass  

**Configuration (LOCKED):**
```python
- Grid: 20×20 with rooms
- Agents: 2
- Episodes: 300
- Steps/episode: 150
- Map: 2-3 rooms (simple obstacles)
- Sensor range: 4
- Comm range: 8.0
```

**Success Criteria (LOCKED):**
```python
✅ QMIX coverage: >75%
✅ Independent coverage: 65-75%
✅ QMIX > Independent (p < 0.05)
✅ Random baseline: 40-55%
✅ Greedy baseline: 60-70%
✅ No negative learning
```

**Implementation Status:** PENDING (awaits Tests 1A + 1B success)

---

## Validation Protocol

### Step 1: Run Tests in Order
```bash
# Test 1A (REQUIRED)
py test_1a_single_agent.py

# Only if 1A passes:
py test_1b_two_agents.py

# Only if 1A + 1B pass:
py test_1c_obstacles.py  # (when implemented)
```

### Step 2: Check Results
```bash
# View results
cat test_results/test_1a_results.txt
cat test_results/test_1b_results.txt
```

### Step 3: Lock or Fix
```
IF all tests pass:
  ✅ LOCK Phase 1 components
  ✅ Document validated architecture
  ✅ Proceed to Phase 2 (scaling)

ELSE:
  ❌ DO NOT PROCEED
  ❌ Document failures in failure log
  ❌ Fix root causes (NOT configs)
  ❌ Re-run tests
```

---

## Failure Log Template

When test fails, document:

```yaml
Test: Test_1A
Date: 2025-11-13
Failure: Random baseline 82% (expected 50-60%)
Root Cause: Random agent using trained policy networks
Fix Applied: Rewrote random to pure uniform action selection
Config Changed: NO (IMMUTABLE)
Test Criteria Changed: NO (IMMUTABLE)
Re-run Result: PASS
```

---

## Phase 2: Scalability (LOCKED)

**DO NOT IMPLEMENT UNTIL PHASE 1 PASSES**

### Test 2A: Four Agents, Medium Grid
- Grid: 25×25
- Agents: 4
- Episodes: 400
- Success: QMIX >70%, beats Independent (p<0.05)

### Test 2B: Six Agents, Large Grid
- Grid: 30×30
- Agents: 6
- Episodes: 500
- Success: QMIX >65%, beats Independent (p<0.05)

---

## Phase 3: Generalization (LOCKED)

**DO NOT IMPLEMENT UNTIL PHASE 1+2 PASS**

### Test 3A: Grid-Agnostic
- Train: 20×20
- Test: 30×30 (unseen)
- Success: Coverage >60% on unseen size

---

## Success Tiers (IMMUTABLE)

### Tier 1: Baseline (Minimum Viable)
```
✅ Tests 1A + 1B + 1C pass
✅ QMIX beats Independent (p<0.05)
✅ Training stable (no divergence)

Claim: "QMIX provides coordination in multi-agent coverage"
Publication: Workshop paper
```

### Tier 2: Strong (Desired)
```
✅ All Tier 1 +
✅ Tests 2A + 2B pass (scales to 4-6 agents)
✅ Performance gap: 4-6% improvement

Claim: "QMIX coordination scales to moderate team sizes"
Publication: Conference paper
```

### Tier 3: Exceptional (Stretch)
```
✅ All Tier 2 +
✅ Test 3A passes (grid-agnostic)
✅ Beats greedy by >10% on complex maps

Claim: "QMIX learns generalizable coordination policies"
Publication: Top-tier conference
```

---

## Current Status

**Phase 1:** IN PROGRESS  
**Test 1A:** NOT RUN  
**Test 1B:** NOT RUN  
**Test 1C:** NOT IMPLEMENTED  

**Phase 2:** LOCKED (awaiting Phase 1)  
**Phase 3:** LOCKED (awaiting Phase 1+2)  

**Project Completion:** ~20% (validated components only)  

---

## Commands Quick Reference

```bash
# Run foundation tests
py test_1a_single_agent.py  # Single agent validation
py test_1b_two_agents.py    # Coordination validation

# Check results
cat test_results/test_1a_results.txt
cat test_results/test_1b_results.txt

# View this document
cat FOUNDATION_VALIDATION.md
```

---

## Commitment

**We will NOT:**
- ❌ Change test criteria when tests fail
- ❌ Adjust configs to make results look better
- ❌ Add features before validating foundations
- ❌ Move to next phase without passing current

**We WILL:**
- ✅ Hold to immutable test criteria
- ✅ Debug failures without changing goals
- ✅ Document all failures honestly
- ✅ Build only on validated foundations

---

**Last Modified:** 2025-11-13  
**Framework Status:** LOCKED  
**Next Action:** Run `py test_1a_single_agent.py`
