# Agents Directory - Documentation Hub

This directory contains all project documentation for the FedCDH implementation.

**Last Updated**: April 14, 2026
**Status**: Hybrid implementation roadmap added

---

## 📋 Active Documentation

### Primary Reference Documents

#### 1. **`working_state.md`** (62K) - **MAIN PROJECT STATE**
**Purpose**: Living document tracking all implementation progress, bug fixes, and current status

**Key Sections**:
- Current Status (achievements, limitations)
- Implementation Overview (architecture, design decisions)
- Critical Bug Fixes & Learnings (Bugs 1-6 documented)
- **Hybrid Mode Rewrite Plan** (April 13, 2026) - Motivation and verification
- Code Quality (cleanup phases 1-3)
- Testing & Validation
- Pre-Thesis Validation Plan

**When to Use**:
- Check current implementation status
- Review bug history and fixes
- Understand architectural decisions
- See what's been validated

---

#### 2. **`hybrid_implementation_roadmap.md`** (53K) - **MASTER'S THESIS IMPLEMENTATION PLAN** ⭐ NEW
**Purpose**: Concrete 3-week roadmap to implement theoretically correct Mixture-then-Product hybrid mode

**Key Sections**:
- **Week 1**: Core Probabilistic Circuit Classes (Days 1-5)
  - GroupMixture, ProductOverGroups, ProductOverGroupsWithOverlap
- **Week 2**: Integration & Validation (Days 6-10)
  - Automatic feature grouping, FedCDH integration, smoke tests
- **Week 3**: Experiments & Documentation (Days 11-15)
  - Sachs experiments, statistical analysis, thesis sections

**Research Context**:
- Based on Master's thesis: "Federated Causal Discovery with Probabilistic Circuits"
- Grounded in Seng et al. (2025) paper verification
- Empirical study (no formal guarantees required)

**Deliverables**:
- 3 new PC classes (~300 lines)
- Sachs experimental results (4 configs × 5 seeds)
- Thesis Methods + Results sections

**When to Use**:
- Starting hybrid mode implementation
- Need step-by-step guide with code examples
- Writing thesis documentation
- Understanding Mixture-then-Product theory

---

#### 3. **`research_guide.md`** (14K)
**Purpose**: Research context, theoretical background, and investigation guidelines

**Key Sections**:
- Research Questions (RQ1-RQ4)
- Key Concepts (SPNs, Federated Learning, Causal Discovery)
- Investigation Strategies
- Critical Files Reference

**When to Use**:
- Understanding research motivation
- Clarifying theoretical concepts
- Planning experiments

---

#### 4. **`thesis_experiments_plan.md`** (25K)
**Purpose**: Comprehensive thesis experiment planning

**Key Sections**:
- Experimental Design
- Datasets (Synthetic, Sachs, Semiconductor)
- Baseline Comparisons
- Metrics and Evaluation

**When to Use**:
- Planning thesis experiments
- Designing benchmarks
- Comparing with baselines

---

#### 5. **`user_habits.md`** (6.2K)
**Purpose**: User preferences, workflow patterns, coding style

**Key Sections**:
- Coding Preferences
- Git Workflow
- Project Structure
- Communication Style

**When to Use**:
- Understanding user expectations
- Following project conventions

---

## 🗂️ Reference Documents

### Paper Verification & Findings

**Location**: `/tmp/paper_verification_findings.md` (created during hybrid analysis)

**Purpose**: Detailed verification of Seng et al. (2025) paper answering 4 critical questions:
1. Is Mixture-then-Product correct? ✅ YES
2. Overlapping features supported? ✅ YES
3. Weight learning method? ⚠️ One-pass, no EM
4. Feature grouping strategy? ✅ Automatic

**Note**: Key findings integrated into `working_state.md` and `hybrid_implementation_roadmap.md`

---

## 📁 Archive

### `archive/` Directory
Contains historical documents preserved for reference:

**Recent Cleanup Reports** (March 31, 2026):
- `FEDCDH_CRITICAL_ANALYSIS.md` (13K) - Deep dive into over-engineering
- `PHASE2_CLEANUP_REPORT.md` (11K) - Feature maps simplification
- `THEORETICAL_VALIDATION_REPORT.md` (17K) - 7 core requirements validation

**Historical Documents**:
- Various bug fixes, meeting notes, early analyses
- All critical information consolidated in active documents

**When to Use**: Need detailed historical context for specific cleanup phases

---

## 🎯 Quick Reference Guide

### I want to...

**...understand current project status**
→ Read `working_state.md`

**...implement hybrid mode for thesis**
→ Follow `hybrid_implementation_roadmap.md` (3-week plan)

**...understand why hybrid needs rewrite**
→ See `working_state.md` > "Hybrid Mode Rewrite" section

**...plan thesis experiments**
→ Check `thesis_experiments_plan.md`

**...understand theoretical foundations**
→ Read `research_guide.md`

**...see what's been validated**
→ Check `working_state.md` > "Testing & Validation" section

**...understand Mixture-then-Product theory**
→ Read `hybrid_implementation_roadmap.md` > "Theoretical Foundation"

**...find paper references**
→ `/agents/reference/` directory has PDFs

---

## 📊 Document Hierarchy

```
agents/
├── README.md (this file)
│
├── PRIMARY REFERENCES
│   ├── working_state.md          [Current status, bug history, validation]
│   └── hybrid_implementation_roadmap.md  [3-week implementation plan] ⭐ NEW
│
├── RESEARCH & PLANNING
│   ├── research_guide.md          [Theoretical background]
│   ├── thesis_experiments_plan.md [Experiment design]
│   └── user_habits.md             [Workflow preferences]
│
├── REFERENCE PAPERS
│   └── reference/
│       ├── FedCDH.pdf (Li et al. 2024)
│       ├── Master Thesis Topic.pdf
│       └── Scaling Probabilistic Circuits via Data Partitioning.pdf (Seng et al. 2025)
│
└── ARCHIVE
    ├── README.md                   [Archive index]
    └── cleanup_reports/            [Detailed cleanup reports]
```

---

## 🔄 Update History

| Date | Update | Details |
|------|--------|---------|
| **April 14, 2026** | Hybrid Implementation Roadmap | Added 53K comprehensive 3-week implementation plan for Mixture-then-Product hybrid mode |
| **April 13, 2026** | Hybrid Rewrite Plan | Added motivation and paper verification to working_state.md |
| **March 31, 2026** | Documentation Consolidation | Merged cleanup reports into working_state.md, organized archive |
| **March 30, 2026** | Critical Bug Fixes | Fixed 2 routing bugs in global SPN |
| **March 25, 2026** | SPN Quality Framework | Added comprehensive evaluation framework |

---

## 🎓 For Master's Thesis Work

**Primary Documents for Thesis**:
1. `hybrid_implementation_roadmap.md` - Implementation guide (START HERE for hybrid work)
2. `working_state.md` - Current status and bug history
3. `thesis_experiments_plan.md` - Experiment design

**Thesis Timeline**:
- **Weeks 1-3**: Implement hybrid mode (follow roadmap)
- **Week 4**: Run Sachs experiments
- **Week 5**: Analysis and thesis writing

**Success Criteria** (from roadmap):
- ✅ Correct Mixture-then-Product implementation
- ✅ Empirical validation on Sachs dataset
- ✅ Hybrid ≠ horizontal/vertical results
- ✅ Methods + Results sections written

---

## 📝 Notes

**Document Philosophy**:
- All critical information in active documents
- Archive preserves detailed historical context
- Living documents updated as project evolves
- Every implementation insight documented immediately

**Before Starting Hybrid Implementation**:
1. Review `hybrid_implementation_roadmap.md` thoroughly
2. Understand motivation in `working_state.md` > "Hybrid Mode Rewrite"
3. Check current status in `working_state.md` > "Current Status"
4. Read Seng et al. (2025) paper in `reference/`

---

**Need Help?** All questions should reference one of the above documents for context.
