# User Habits

## Learned Preferences and Patterns

### Coding Style
- **Modular Design**: Clear separation of concerns (FedCDH, FedPC, CI tests, orientation in separate modules)
- **Comprehensive Docstrings**: Detailed function-level documentation with parameter descriptions
- **Type Hints**: Uses type annotations in function signatures
- **Naming Conventions**:
  - Classes: PascalCase (e.g., `FedCDH`, `LocalSPNWrapper`, `SimulatedFederatedKMeans`)
  - Functions: snake_case (e.g., `cdnod_alg`, `compute_mechanism_variance`)
  - Constants: UPPER_SNAKE_CASE for configs (e.g., `PRODUCTION_CONFIGS`)
- **Code Organization**:
  - Core algorithms in `causallearn/` library
  - Experiments in `tests/benchmarks/`
  - Utilities in `causallearn/utils/`
- **Dependencies**: Prefers standard scientific Python stack (torch, numpy, scipy, pandas, matplotlib, seaborn)

### Git Commit Style
**Pattern**: Conventional Commits with descriptive bodies

**Prefixes Used**:
- `feat:` - New features (e.g., "feat: add L1 sparsity to FedPC")
- `fix:` - Bug fixes (e.g., "fix: resolve dimension mismatch")
- `test:` - Test additions or modifications (e.g., "test: add comprehensive scalability benchmark")
- `refactor:` - Code restructuring without behavior change (e.g., "refactor: clean up remaining unused imports")
- `docs:` - Documentation changes (e.g., "docs: update README")

**Structure**:
```
<type>: <short summary (imperative, <50 chars)>

[Optional detailed bullet points with - prefix]
- First change
- Second change
- Third change
```

**Examples from Recent Commits**:
1. Short, focused commits:
   - `feat: add L1 sparsity to FedPC and optimize SPN_CIT performance`
   - `test: add comprehensive scalability benchmark for FedSPN vs KCI`

2. Multi-component commits with detailed bodies:
   ```
   Refactor and optimize FedCDH benchmarking pipeline:
   - Consolidate logic into causallearn library
   - Fix bug in mechanism invariance orientation logic
   - Enhance plotting with 3-panel visualization
   - Optimize SPN-CIT with vectorization
   ```

**Principles**:
- Focus on "what" and "why", not "how"
- Use imperative mood ("add" not "added", "fix" not "fixed")
- Keep subject line under 70 characters
- Use bullet points for multi-part changes
- Avoid unnecessary words like "complex" or "risk"

### Interaction Style
- **Professional and Direct**: Prefers concise, technical communication
- **Markdown Formatting**: Uses headers, bullet points, code blocks, and tables
- **Research-Oriented**: Values theoretical foundations and experimental rigor
- **Systematic Approach**: Appreciates structured planning (hence the `agents/` memory bank)

### Project Workflow

#### Branch Strategy
- **Main Branch**: `main` - stable, production-ready code
- **Feature Branch**: `fedpc` - active development branch for FedPC integration
- Merge to main after validation

#### Experiment Workflow
1. Configure experiments in `tests/benchmarks/configs.py`
2. Execute via `tests/benchmarks/run_experiment.py`
3. Results stored in `tests/experiments/<config_name>_<dataset>_<params>_<seed>_<timestamp>/`
4. Aggregate with `tests/benchmarks/analyze_results.py`
5. Generate publication-ready plots

#### Testing Strategy
- **Smoke Tests**: Quick validation (e.g., `test_fedcdh_simple.py` with 3-node synthetic)
- **Unit Tests**: Module-level tests in `tests/unit/`
- **Benchmark Suite**: Full pipeline tests with real datasets
- **Scalability Tests**: Parameter grid searches with `benchmark_scalability.py`

### Tool Preferences
- **Development**: PyCharm (inferred from project path)
- **Version Control**: Git with conventional commits
- **Package Management**: Standard `setup.py` (inferred)
- **Visualization**: Matplotlib + Seaborn for publication-quality plots
- **Notebooks**: Uses `.ipynb` files (evidence from NotebookEdit in tools)

### Communication Preferences
- **Documentation Updates**: Prefers in-code docstrings over external docs
- **Planning**: Values upfront design discussion before implementation
- **Validation**: Expects clear explanation before risky changes (refactors, dependency changes)
- **Commit Hygiene**: Never stage/commit unless explicitly requested

### Common Commands
Based on project structure and workflows:
```bash
# Development
python setup.py install              # Install package
python -m pytest tests/              # Run test suite

# Experiments
python tests/benchmarks/run_experiment.py --config fedspn_horizontal --seed 0
python tests/benchmarks/benchmark_scalability.py

# Quick validation
python test_fedcdh_simple.py        # Smoke test

# Analysis
python tests/benchmarks/analyze_results.py tests/experiments/
```

### Workflow Constraints
- **NO auto-committing**: Never stage or commit changes unless explicitly asked
- **NO auto-reverting**: Never revert changes unless requested
- **Explain risky changes**: Always explain plan before refactors, dependency changes, or structural modifications
- **Agent Memory System**: Store all agent reference files in `/agents` directory
  - Keep only 4 core files: research_guide.md, user_habits.md, working_state.md, thesis_experiments_plan.md
  - Consolidate technical discussions into working_state.md chronologically
  - Move analysis documents to /agents before archiving
- **Validate before merging**: Run smoke tests and benchmarks before merging to main
- **Documentation Policy** (Added March 25, 2026):
  - All exchanges documented in /agents/ active md files ONLY
  - **DO NOT create new md files** unless explicitly requested by user
  - **Only update files if information is CRITICAL** (bug fixes, major decisions, meeting outcomes)
  - Keep project repository minimalistically clean
  - Avoid creating session summaries, temporary docs, or analysis files
  - If documentation needed, integrate into one of the 4 core files

### Research Habits (Inferred from Code)
- **Rigorous Baselines**: Compares against multiple methods (FisherZ, KCI, voting)
- **Monte Carlo Evaluation**: Runs experiments with multiple random seeds (5-10)
- **Publication Focus**: Generates plots with error bars, clean formatting, and parameter metadata
- **Reproducibility**: Logs all hyperparameters, configs, and random seeds
- **Theoretical Grounding**: Implements novel methods (mechanism invariance) with clear mathematical principles

---
*Created on 2026-03-06 | Updated by Claude Code*
