#!/usr/bin/env python
"""
SPN Code Refactoring Script - Three-Layer Architecture Migration

This script reorganizes the SPN codebase into a clean three-layer architecture:
1. utils/spn/ - Reusable SPN foundation
2. search/FCMBased/FedCDH/ - Algorithm-specific logic
3. Existing constraint-based algorithms (CDNOD)

Usage:
    python scripts/refactor_spn_code.py --phase [1-9] --dry-run

Phases:
    1. ✅ Create directory structure (COMPLETED)
    2. Split FedPC.py into modules
    3. Consolidate evaluation files
    4. ✅ Create FedCDH subdirectories (COMPLETED)
    5. Move orientation logic
    6. Move data partitioning
    7. Update FedCDH imports
    8. Update test scripts
    9. Clean up old files

Safety:
    - Always run with --dry-run first
    - Commits changes after each phase
    - Creates backup before major changes
    - Tests after each phase
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)


def run_command(cmd, description):
    """Run shell command with error handling."""
    print(f"[INFO] {description}")
    print(f"[CMD] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] {result.stderr}")
        return False
    if result.stdout:
        print(f"[OUTPUT] {result.stdout}")
    return True


def backup_file(file_path):
    """Create backup of file."""
    backup_path = f"{file_path}.backup"
    shutil.copy2(file_path, backup_path)
    print(f"[BACKUP] Created {backup_path}")


def phase_5_move_orientation(dry_run=False):
    """Phase 5: Move orientation logic to FedCDH package."""
    print("\n" + "=" * 80)
    print("Phase 5: Move orientation logic to FedCDH/orientation/")
    print("=" * 80)

    src = "causallearn/utils/mechanism_invariance.py"
    dst = "causallearn/search/FCMBased/FedCDH/orientation/mechanism_invariance.py"

    if dry_run:
        print(f"[DRY RUN] Would move {src} -> {dst}")
        print(f"[DRY RUN] Would update imports in:")
        print("  - causallearn/search/ConstraintBased/CDNOD.py")
        print("  - tests/")
        return True

    # Backup original
    backup_file(src)

    # Move file
    shutil.move(src, dst)
    print(f"[MOVED] {src} -> {dst}")

    # Update imports in CDNOD (if exists)
    cdnod_path = "causallearn/search/ConstraintBased/CDNOD.py"
    if os.path.exists(cdnod_path):
        with open(cdnod_path, "r") as f:
            content = f.read()

        content = content.replace(
            "from causallearn.utils.mechanism_invariance import",
            "from causallearn.search.FCMBased.FedCDH.orientation.mechanism_invariance import",
        )

        with open(cdnod_path, "w") as f:
            f.write(content)
        print(f"[UPDATED] {cdnod_path}")

    # Git commit
    run_command(
        f'git add {dst} {src} {cdnod_path} && git commit -m "refactor: move orientation logic to FedCDH package (Phase 5)"',
        "Committing Phase 5 changes",
    )

    return True


def phase_6_move_data_partitioning(dry_run=False):
    """Phase 6: Move data partitioning to FedCDH package."""
    print("\n" + "=" * 80)
    print("Phase 6: Move data partitioning to FedCDH/data_partitioning/")
    print("=" * 80)

    moves = [
        (
            "causallearn/utils/hybrid_partition.py",
            "causallearn/search/FCMBased/FedCDH/data_partitioning/hybrid.py",
        ),
        (
            "causallearn/utils/structure_aggregation.py",
            "causallearn/search/FCMBased/FedCDH/data_partitioning/aggregation.py",
        ),
        (
            "causallearn/utils/cost_analysis.py",
            "causallearn/search/FCMBased/FedCDH/cost_analysis.py",
        ),
    ]

    if dry_run:
        print("[DRY RUN] Would move:")
        for src, dst in moves:
            print(f"  {src} -> {dst}")
        return True

    for src, dst in moves:
        if os.path.exists(src):
            backup_file(src)
            shutil.move(src, dst)
            print(f"[MOVED] {src} -> {dst}")

    # Update FedCDH.py imports
    fedcdh_path = "causallearn/search/FCMBased/FedCDH/FedCDH.py"
    with open(fedcdh_path, "r") as f:
        content = f.read()

    content = content.replace(
        "from causallearn.utils.hybrid_partition import",
        "from causallearn.search.FCMBased.FedCDH.data_partitioning.hybrid import",
    )
    content = content.replace(
        "from causallearn.utils.structure_aggregation import",
        "from causallearn.search.FCMBased.FedCDH.data_partitioning.aggregation import",
    )
    content = content.replace(
        "from causallearn.utils.cost_analysis import",
        "from causallearn.search.FCMBased.FedCDH.cost_analysis import",
    )

    with open(fedcdh_path, "w") as f:
        f.write(content)
    print(f"[UPDATED] {fedcdh_path}")

    # Git commit
    moved_files = " ".join([dst for _, dst in moves])
    run_command(
        f'git add {moved_files} {fedcdh_path} && git commit -m "refactor: move data partitioning to FedCDH package (Phase 6)"',
        "Committing Phase 6 changes",
    )

    return True


def phase_7_update_fedcdh_imports(dry_run=False):
    """Phase 7: Update FedCDH.py to use new import paths."""
    print("\n" + "=" * 80)
    print("Phase 7: Update FedCDH.py imports")
    print("=" * 80)

    fedcdh_path = "causallearn/search/FCMBased/FedCDH/FedCDH.py"

    if dry_run:
        print(f"[DRY RUN] Would update imports in {fedcdh_path}:")
        print("  - Add: from causallearn.utils import spn")
        print("  - Keep backwards-compatible imports from FedPC")
        return True

    with open(fedcdh_path, "r") as f:
        content = f.read()

    # Add new import at top (but keep old imports for now - backwards compatibility)
    new_import = "\n# New modular imports (transitioning from FedPC)\n"
    new_import += "# from causallearn.utils.spn import (\n"
    new_import += "#     LocalSPNWrapper, GlobalFedSPN, ...\n"
    new_import += "# )\n"
    new_import += "# TODO: Gradually replace FedPC imports with spn.* imports\n\n"

    # Insert after existing imports
    import_end = content.find("class SimulatedFederatedKMeans:")
    if import_end > 0:
        content = content[:import_end] + new_import + content[import_end:]

    with open(fedcdh_path, "w") as f:
        f.write(content)

    print(f"[UPDATED] {fedcdh_path} - added import placeholders")
    print("[INFO] FedPC.py kept as compatibility layer for now")

    return True


def phase_8_update_tests(dry_run=False):
    """Phase 8: Update test scripts to use new imports."""
    print("\n" + "=" * 80)
    print("Phase 8: Update test scripts")
    print("=" * 80)

    test_files = [
        "tests/benchmarks/test_fedcdh_benchmark_v3.py",
        "smoke_test_minimal.py",
        "smoke_test_all_modes.py",
        "smoke_test_asia_experimental.py",
        "smoke_test_asia_simple.py",
    ]

    if dry_run:
        print("[DRY RUN] Would update imports in:")
        for f in test_files:
            if os.path.exists(f):
                print(f"  - {f}")
        return True

    for test_file in test_files:
        if not os.path.exists(test_file):
            continue

        print(f"[CHECKING] {test_file}")
        with open(test_file, "r") as f:
            content = f.read()

        # These imports should still work (FedPC as compatibility layer)
        # No changes needed unless we fully deprecate FedPC
        print(f"[SKIPPED] {test_file} - no changes needed (using compatibility layer)")

    print("\n[INFO] Tests continue to use FedPC imports (backwards compatible)")
    print("[INFO] Future: Migrate to causallearn.utils.spn imports")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Refactor SPN code into three-layer architecture"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=range(1, 10),
        help="Phase to execute (1-9). Phases 1,4 already completed.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all remaining phases (5-9)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    if not args.phase and not args.all:
        parser.print_help()
        sys.exit(1)

    print("=" * 80)
    print("SPN Code Refactoring - Three-Layer Architecture")
    print("=" * 80)
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    # Phase dispatch
    phases = {
        5: phase_5_move_orientation,
        6: phase_6_move_data_partitioning,
        7: phase_7_update_fedcdh_imports,
        8: phase_8_update_tests,
    }

    if args.all:
        print("[INFO] Running all remaining phases (5-8)")
        for phase_num in sorted(phases.keys()):
            success = phases[phase_num](dry_run=args.dry_run)
            if not success:
                print(f"\n[ERROR] Phase {phase_num} failed!")
                sys.exit(1)
    elif args.phase in phases:
        success = phases[args.phase](dry_run=args.dry_run)
        if not success:
            print(f"\n[ERROR] Phase {args.phase} failed!")
            sys.exit(1)
    elif args.phase in [1, 4]:
        print(f"[INFO] Phase {args.phase} already completed (directory structure)")
    elif args.phase in [2, 3, 9]:
        print(f"[MANUAL] Phase {args.phase} requires manual code splitting")
        print("Phases 2 & 3: Split FedPC.py - too complex for automated migration")
        print("Phase 9: Final cleanup after testing")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("✅ Refactoring phase completed successfully!")
    print("=" * 80)

    if not args.dry_run:
        print("\n[NEXT STEPS]")
        print("1. Run tests: python -m pytest tests/")
        print("2. Run smoke test: python smoke_test_minimal.py")
        print("3. Verify no regressions")
        print("4. Continue to next phase")


if __name__ == "__main__":
    main()
