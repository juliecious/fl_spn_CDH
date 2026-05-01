# Agent Documentation Directory

This directory contains documentation for the FedCDH v2 implementation.

## Current Documentation

### Core Documentation

1. **working_state.md** - Main development log and current status
   - Implementation history
   - Bug fixes log
   - v3 experiment plan
   - Complete changelog

2. **BUG_FIXES_CONSOLIDATED.md** - All bug fixes
   - Context column bugs (2026-05-01)
   - GPU sampling dimension fix
   - Hybrid mode fixes
   - Evaluation directory fix

3. **TEST_RESULTS_CONSOLIDATED.md** - All test results
   - CPU smoke test results
   - GPU simulation results
   - Verification tests
   - Performance benchmarks

### Historical Documentation

4. **V1_VS_V2_CHANGES.md** - Detailed comparison of v1 vs v2
   - Line-by-line changes
   - Feature additions
   - Architecture changes

5. **V2_GPU_READINESS_SUMMARY.md** - GPU deployment readiness (2026-04-30)
   - Pre-deployment checklist
   - Known issues at that time
   - Testing results

6. **HYBRID_VERIFICATION_GUIDE.md** - Hybrid mode verification
   - Sum-over-products validation
   - Test procedures

### Implementation Guides

7. **hybrid_implementation_roadmap.md** - Hybrid mode implementation details
   - Complete implementation plan
   - Code examples
   - Design decisions

8. **research_guide.md** - Research methodology
9. **thesis_experiments_plan.md** - Thesis experiment planning
10. **user_habits.md** - User preferences and conventions

### Deprecated/Historical

11. **fix.md** - Old fix log (consolidated into BUG_FIXES_CONSOLIDATED.md)

## Reference Materials

Located in `agents/reference/`:
- FedCDH.pdf - Li et al. 2024 paper
- Scaling Probabilistic Circuits via Data Partitioning.pdf - Seng et al. 2025
- Master Thesis Topic.pdf - Thesis topic description

## Quick Navigation

**Current Status**: See `working_state.md`
**Bug Fixes**: See `BUG_FIXES_CONSOLIDATED.md`
**Test Results**: See `TEST_RESULTS_CONSOLIDATED.md`
**Implementation Details**: See `hybrid_implementation_roadmap.md`
**Deployment**: See `V2_GPU_READINESS_SUMMARY.md`

## Document Maintenance

- **Active**: working_state.md, BUG_FIXES_CONSOLIDATED.md, TEST_RESULTS_CONSOLIDATED.md
- **Reference**: V1_VS_V2_CHANGES.md, V2_GPU_READINESS_SUMMARY.md, hybrid_implementation_roadmap.md
- **Historical**: fix.md (superseded by BUG_FIXES_CONSOLIDATED.md)

Last updated: 2026-05-01
