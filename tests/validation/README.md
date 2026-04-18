# Validation Scripts

This directory contains validation and debugging scripts for FedCDH implementation.

## Scripts

### Verification Scripts

**`verify_federated_compliance.py`**
- Verifies implementation complies with federated learning principles
- Checks: Data partitioning, local training, parameter sharing
- Output: Compliance report for H/V/Hybrid modes
- Run: `python tests/validation/verify_federated_compliance.py`

**`verify_hybrid_fix.py`**
- Verifies hybrid mode evaluation fix is applied
- Checks: X_aug_global_train storage and usage
- Output: Fix implementation confirmation
- Run: `python tests/validation/verify_hybrid_fix.py`

### Debug Scripts

**`debug_vertical_ll.py`**
- Diagnostic tool for vertical mode log-likelihood issues
- Tests: FederatedProduct correctness, feature extraction
- Created: April 17, 2026 (Investigation)
- Run: `python tests/validation/debug_vertical_ll.py`

**`debug_vertical_umap.py`**
- Diagnostic tool for vertical mode UMAP generation
- Tests: Local SPN evaluation with feature subsets
- Created: April 13, 2026
- Run: `python tests/validation/debug_vertical_umap.py`

## Purpose

These scripts were created during the vertical/hybrid mode investigation
(April 13-18, 2026) to diagnose and fix evaluation data mismatch issues.

Key findings documented in `agents/working_state.md`.
