# Agent Memory System - Structure & Usage

**Updated**: March 15, 2026
**Purpose**: Reference guide for Claude agent memory system

---

## Core Files (Keep Updated)

### 1. **research_guide.md** - Thesis Roadmap
**Purpose**: High-level research strategy and design decisions
**Contents**:
- Research questions (RQ1-RQ5)
- Design decisions (confirmed & open)
- Novel contributions (vertical regularization hypothesis)
- Evaluation strategy (datasets, baselines, metrics)
- Thesis structure outline
- Open questions & future work

**Update when**: Research direction changes, new insights emerge

---

### 2. **user_habits.md** - Workflow Conventions
**Purpose**: Coding style, git patterns, interaction preferences
**Contents**:
- Coding style (naming, structure, dependencies)
- Git commit patterns (conventional commits)
- Workflow constraints (no auto-commit, agent files in /agents)
- Testing strategy (smoke → benchmark → scalability)
- Common commands

**Update when**: New patterns identified, workflow changes

---

### 3. **working_state.md** - Chronological Progress
**Purpose**: Live status of implementation, tasks, and conversations
**Contents**:
- Current project state (branch, status, last updated)
- Implementation status (completed, in progress, known issues)
- Active TODOs with priority (🔴/🟡/🟢/🔵)
- Recently completed work (dated entries)
- Commit history with descriptions

**Update after**: Each work session, major commits, task completion

**Format for session entries**:
```markdown
#### YYYY-MM-DD - Session Title
- **Topic 1**: Description and outcome
  - Detail 1
  - Detail 2
- **Topic 2**: Description and outcome
- **Commit XXXXXXX**: Changes made
```

---

### 4. **thesis_experiments_plan.md** - Experiment Execution
**Purpose**: Detailed experiment plan for publication-ready results
**Contents**:
- Timeline overview (4-week MVP plan)
- Phase 1: Sachs core benchmark (50 runs)
- Phase 2: Scalability analysis (90 runs)
- Phase 3: Ablation studies (30 runs)
- Execution commands and troubleshooting
- Success metrics and risk mitigation

**Update when**: Experiment plan changes, phases complete, risks materialize

---

## Supporting Files (Reference)

### Technical Analysis (Current Session)
- **TECHNICAL_RECOMMENDATIONS_TASKS.md**: 11 executable tasks with prompts
- **FEDERATED_CONCURRENCY_ANALYSIS.md**: Parallelization analysis vs Jonas Seng
- **SUMMARY_TECHNICAL_DELIVERABLES.md**: Quick reference for March 15 session

### Archive (Historical)
- **archive/**: Historical fix documentation (CRITICAL_FIXES_MAR9.md, GPU_DEVICE_FIX.md, etc.)
- Keep for reference but don't update

---

## Agent File Conventions (user_habits.md)

### Storage Location
- ✅ **All agent files**: Store in `/agents` directory
- ❌ **Root directory**: Only project files (README.md, setup.py, etc.)

### File Lifecycle
1. **Active work**: Create in `/agents` with descriptive name
2. **Session end**: Consolidate into one of 4 core files
3. **Historical**: Move to `/agents/archive/` if no longer actively referenced

### Naming Conventions
- **Core files**: lowercase with underscores (research_guide.md)
- **Session files**: UPPERCASE with descriptive names (TECHNICAL_RECOMMENDATIONS_TASKS.md)
- **Archive files**: Keep original names, move to archive/ subdirectory

---

## Usage Patterns

### Starting a Session
1. Read **working_state.md** to understand current status
2. Check **Active TODOs** for pending work
3. Reference **research_guide.md** for context on design decisions

### During a Session
1. Track decisions and progress mentally
2. Reference **user_habits.md** for coding/git conventions
3. Create session-specific files in `/agents` as needed

### Ending a Session
1. Update **working_state.md**:
   - Add dated entry to "Recently Completed"
   - Update "Active TODOs" status
   - Update "Last Updated" date
2. Consolidate technical discussions into working_state.md
3. Archive or delete session-specific files no longer needed
4. Update other core files if research direction changed

### Before Major Milestones (Experiments, Thesis Defense)
1. Review all 4 core files for consistency
2. Update **thesis_experiments_plan.md** with latest timeline
3. Archive completed tasks from **working_state.md**

---

## Current State (March 15, 2026)

### Core Files Status
- ✅ **research_guide.md**: Up to date (includes vertical regularization hypothesis)
- ✅ **user_habits.md**: Updated with agent file conventions
- ✅ **working_state.md**: Updated with March 15 comprehensive review
- ✅ **thesis_experiments_plan.md**: Up to date (170-run plan ready)

### Supporting Files
- ✅ **TECHNICAL_RECOMMENDATIONS_TASKS.md**: 11 tasks ready for execution
- ✅ **FEDERATED_CONCURRENCY_ANALYSIS.md**: Complete parallelization analysis
- ✅ **SUMMARY_TECHNICAL_DELIVERABLES.md**: Session summary

### Archived
- CRITICAL_FIXES_MAR9.md, GPU_DEVICE_FIX.md, VERTICAL_SCENARIO_FIX.md (moved to archive/)
- session_summary.md, README.md (consolidated into working_state.md, deleted)

### Next Actions
1. Execute TASK-1, TASK-2, TASK-3 (HIGH priority) this week
2. Begin Phase 1 experiments (50 runs on GPU)
3. Update working_state.md after task completion

---

## Maintenance Guidelines

### What to Keep in Core Files
- ✅ High-level strategy and decisions (research_guide.md)
- ✅ Current status and active tasks (working_state.md)
- ✅ Coding conventions and patterns (user_habits.md)
- ✅ Detailed experiment plans (thesis_experiments_plan.md)

### What to Archive
- ❌ Completed bug fix analysis (once fixed and validated)
- ❌ Session-specific technical discussions (after consolidation)
- ❌ Historical meeting notes (after action items moved to TODOs)

### What to Delete
- ❌ Duplicate information
- ❌ Outdated session summaries (after consolidation)
- ❌ Temporary analysis files (after integration into core files)

---

## Quick Reference Commands

### View Current State
```bash
cat agents/working_state.md | head -50        # Current status
grep "🔴\|🟡\|🟢" agents/working_state.md      # Active TODOs
tail -100 agents/working_state.md             # Recent work
```

### Update After Session
```bash
# Add dated entry to working_state.md
# Update "Last Updated" field
# Move completed tasks to "Completed TODOs" section
```

### Check Experiment Status
```bash
cat agents/thesis_experiments_plan.md | grep "Phase"
cat agents/working_state.md | grep "IMPL-2"
```

---

*This file provides structure for the agent memory system.*
*Update this summary if conventions change.*
