# Sync Files to GPU Server

## Problem
The new `causallearn/utils/spn/core/` and other subdirectories exist locally but not on the server.

## Solution Options

### Option 1: Git Push/Pull (Recommended)

On your **local machine**:
```bash
cd /Users/M279402/PycharmProjects/fl_spn_CDH

# Stage all new SPN files
git add causallearn/utils/spn/core/
git add causallearn/utils/spn/causal/
git add causallearn/utils/spn/federated/client_init.py
git add causallearn/utils/spn/federated/structure_sync.py
git add causallearn/utils/spn/federated/latent_routing.py

# Commit
git commit -m "feat: add bug fixes and new SPN structure (core, causal, client_init)"

# Push to remote
git push origin v3-comprehensive-fixes
```

On the **GPU server**:
```bash
cd /home/fang/fedcdh_pc

# Pull latest changes
git fetch origin
git checkout v3-comprehensive-fixes
git pull origin v3-comprehensive-fixes

# Reinstall package
pip install -e . --force-reinstall --no-deps

# Verify
python -c "from causallearn.utils.spn.core import LocalSPNWrapper; print('✓ Import successful')"
```

---

### Option 2: Direct rsync (If Git Not Available)

On your **local machine**:
```bash
# Sync the entire causallearn/utils/spn directory
rsync -avz --progress \
  /Users/M279402/PycharmProjects/fl_spn_CDH/causallearn/utils/spn/ \
  fang@cda-server-3:/home/fang/fedcdh_pc/causallearn/utils/spn/

# Also sync the test files
rsync -avz --progress \
  /Users/M279402/PycharmProjects/fl_spn_CDH/tests/benchmarks/test_bug_fixes_dryrun.py \
  fang@cda-server-3:/home/fang/fedcdh_pc/tests/benchmarks/

rsync -avz --progress \
  /Users/M279402/PycharmProjects/fl_spn_CDH/run_asia_horizontal.py \
  fang@cda-server-3:/home/fang/fedcdh_pc/
```

Then on the **GPU server**:
```bash
cd /home/fang/fedcdh_pc

# Reinstall package
pip install -e . --force-reinstall --no-deps

# Verify
python -c "from causallearn.utils.spn.core import LocalSPNWrapper; print('✓ Import successful')"
```

---

### Option 3: Manual File Check (Debug)

On the **GPU server**, run this to verify what's missing:
```bash
cd /home/fang/fedcdh_pc

# Check directory structure
echo "=== Current SPN structure ==="
ls -la causallearn/utils/spn/

echo -e "\n=== Core directory exists? ==="
ls -la causallearn/utils/spn/core/ 2>&1

echo -e "\n=== Causal directory exists? ==="
ls -la causallearn/utils/spn/causal/ 2>&1

echo -e "\n=== New files in federated? ==="
ls -la causallearn/utils/spn/federated/ | grep -E "(client_init|structure_sync|latent_routing)"

# If directories are missing, you need to sync from local machine
```

---

## Quick Verification Script

After syncing, run this on the **GPU server**:

```bash
cd /home/fang/fedcdh_pc

# Create verification script
cat > verify_structure.py << 'EOF'
import sys
import os

print("=== Checking SPN Package Structure ===\n")

# Check directories
dirs_to_check = [
    'causallearn/utils/spn/core',
    'causallearn/utils/spn/causal',
    'causallearn/utils/spn/federated',
    'causallearn/utils/spn/evaluation',
    'causallearn/utils/spn/structure',
]

for dir_path in dirs_to_check:
    exists = os.path.isdir(dir_path)
    status = "✓" if exists else "✗"
    print(f"{status} {dir_path}")

    if exists:
        init_file = os.path.join(dir_path, '__init__.py')
        has_init = os.path.isfile(init_file)
        init_status = "✓" if has_init else "✗"
        print(f"  {init_status} __init__.py")

# Check critical files
print("\n=== Checking Critical Files ===\n")
files_to_check = [
    'causallearn/utils/spn/core/local.py',
    'causallearn/utils/spn/causal/ci_testing.py',
    'causallearn/utils/spn/federated/client_init.py',
]

for file_path in files_to_check:
    exists = os.path.isfile(file_path)
    status = "✓" if exists else "✗"
    print(f"{status} {file_path}")

# Try imports
print("\n=== Testing Imports ===\n")
try:
    from causallearn.utils.spn.core import LocalSPNWrapper
    print("✓ from causallearn.utils.spn.core import LocalSPNWrapper")
except ImportError as e:
    print(f"✗ LocalSPNWrapper import failed: {e}")
    sys.exit(1)

try:
    from causallearn.utils.spn.causal.ci_testing import greedy_dag_search_via_circuit_ci
    print("✓ from causallearn.utils.spn.causal.ci_testing import greedy_dag_search_via_circuit_ci")
except ImportError as e:
    print(f"✗ CI testing import failed: {e}")
    sys.exit(1)

try:
    from causallearn.utils.spn.federated.client_init import initialize_heterogeneous_clients
    print("✓ from causallearn.utils.spn.federated.client_init import initialize_heterogeneous_clients")
except ImportError as e:
    print(f"✗ client_init import failed: {e}")
    sys.exit(1)

print("\n=== All Checks Passed! ===")
EOF

python verify_structure.py
```

---

## Expected Output After Successful Sync

```
=== Checking SPN Package Structure ===

✓ causallearn/utils/spn/core
  ✓ __init__.py
✓ causallearn/utils/spn/causal
  ✓ __init__.py
✓ causallearn/utils/spn/federated
  ✓ __init__.py
✓ causallearn/utils/spn/evaluation
  ✓ __init__.py
✓ causallearn/utils/spn/structure
  ✓ __init__.py

=== Checking Critical Files ===

✓ causallearn/utils/spn/core/local.py
✓ causallearn/utils/spn/causal/ci_testing.py
✓ causallearn/utils/spn/federated/client_init.py

=== Testing Imports ===

✓ from causallearn.utils.spn.core import LocalSPNWrapper
✓ from causallearn.utils.spn.causal.ci_testing import greedy_dag_search_via_circuit_ci
✓ from causallearn.utils.spn.federated.client_init import initialize_heterogeneous_clients

=== All Checks Passed! ===
```

---

## Recommended: Use Git (Option 1)

**Why?**
- Version controlled
- Easier to track changes
- Can rollback if needed
- Clean and reproducible

After syncing via git, run the benchmark:
```bash
nohup python -m tests.benchmarks.test_fedcdh_benchmark_v3 \
  --datasets asia \
  --methods fedspn_h,fedspn_v,fedspn_hy \
  --seeds 42 \
  --K 3 \
  --device cuda \
  --output-dir eval/v3_asia_gpu_test \
  --save-graphs \
  2>&1 | tee test.log &
```
