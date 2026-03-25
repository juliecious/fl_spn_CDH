# Meeting Executive Summary - 1 Page

**Meeting**: Thesis Discussion with Jonas (FedCDH author) & Prof. Devendra Singh Dhami
**Date**: March 24, 2026

---

## What I've Done

✅ **Full FedCDH Implementation** (March 6-15)
- 3 scenarios (H/V/Hy) working
- SPN-based CI testing with permutation tests
- Mechanism invariance orientation
- 800-line comprehensive experiment suite

✅ **Critical Bug Fixed** (March 24)
- Problem: `num_permutations=0` used wrong Chi-squared test
- Fix: Changed to `num_permutations=50` (proper permutation test)
- Result: F1_skeleton improved from 0.4 → **1.0** on validation data

✅ **Key Finding**: Vertical partitioning improves oriented F1 (0.200 vs 0.000)
- Hypothesis: Feature separation acts as regularization
- **Potential novel contribution for thesis**

---

## Critical Questions for Jonas

1. **SPN Performance**: What F1 scores did FedCDH achieve with KCI on Sachs?
2. **Vertical Effect**: Did you observe vertical → better orientation?
3. **Novelty Check**: Is vertical regularization documented in your work?
4. **Realistic Targets**: What F1 should I expect for N=856 (vs your N=5000)?
5. **Implementation**: Does our MI orientation match your paper's method?

---

## Critical Questions for Prof. Dhami

1. **Statistical Validity**: Is 50 permutations standard for α=0.05 testing?
2. **Theoretical Grounding**: Is "vertical as regularization" causally plausible?
3. **Evaluation**: Are our metrics (F1, SHD, comm cost) sufficient?
4. **Privacy**: What privacy level does model parameter sharing provide?
5. **Scope**: Is 6 experiments enough for Master's thesis?

---

## Current Status

**Timeline**: April 30 submission (5 weeks remaining)

| Phase | Status |
|-------|--------|
| Implementation | ✅ Done |
| Bug fixing | ✅ Done (just validated) |
| Synthetic experiments | ⏳ Ready to run |
| Sachs validation | ⏳ Pending |
| Writing | ⏳ 2 weeks budgeted |

**Bottleneck**: Need to validate d=8,K=3 works after bug fix, then run full suite

---

## What I Need from This Meeting

**Minimum**:
1. Validation that thesis scope is sufficient
2. Realistic performance targets (F1 on Sachs)
3. Confirmation vertical effect is novel (or pivot if not)

**Ideal**:
4. Specific implementation feedback
5. Suggestions for strengthening novelty
6. Potential follow-up collaboration

---

## 30-Second Pitch

> "I'm implementing Federated Causal Discovery with SPNs instead of kernel methods. After fixing a critical bug in the statistical test (chi-squared → permutation), accuracy jumped to F1=1.0 on validation. The interesting finding: vertical partitioning improves orientation quality - possibly as regularization. Need to validate this is novel and run comprehensive experiments."

---

## Red Flags to Watch

- ❌ "This exists in our supplementary materials" → need new novelty
- ❌ "Timeline too tight" → reduce scope
- ❌ "Implementation wrong" → major debugging
- ❌ "Results don't make sense" → more investigation

---

## Next Actions (Based on Meeting)

**If scope validated**:
→ Run full suite (6 experiments) + Sachs + write

**If novelty confirmed**:
→ Focus on vertical regularization experiments

**If issues found**:
→ Debug/fix, then pivot research question

---

**Prepared**: March 24, 2026 | **Full doc**: `MEETING_PREPARATION_JONAS_DEVENDRA.md`
