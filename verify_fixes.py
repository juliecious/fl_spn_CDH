#!/usr/bin/env python
"""Quick verification that fixes are applied correctly."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

print("=" * 80)
print("VERIFYING FIXES")
print("=" * 80)
print()

# Check 1: num_permutations default
print("1. Checking num_permutations default in FedCDH.py...")
with open("causallearn/search/FCMBased/FedCDH/FedCDH.py", "r") as f:
    content = f.read()
    if 'num_permutations = getattr(self.args, "num_permutations", 50)' in content:
        print("   ✅ num_permutations default = 50")
    else:
        print("   ❌ num_permutations default NOT set to 50")

# Check 2: numpy array handling in orientation
print("\n2. Checking numpy array handling in mechanism_invariance.py...")
with open(
    "causallearn/search/FCMBased/FedCDH/orientation/mechanism_invariance.py", "r"
) as f:
    content = f.read()
    if "isinstance(client_features, np.ndarray)" in content:
        print("   ✅ Numpy array handling added")
    else:
        print("   ❌ Numpy array handling NOT found")

# Check 3: depth_limit parameter
print("\n3. Checking depth_limit parameter in SkeletonDiscovery.py...")
with open("causallearn/utils/PCUtils/SkeletonDiscovery.py", "r") as f:
    content = f.read()
    if "depth_limit: int | None = None" in content:
        print("   ✅ depth_limit parameter added")
    else:
        print("   ❌ depth_limit parameter NOT found")

    if "if depth_limit is not None and depth > depth_limit:" in content:
        print("   ✅ depth_limit check in while loop")
    else:
        print("   ❌ depth_limit check NOT found")

# Check 4: depth_limit in CDNOD
print("\n4. Checking depth_limit in CDNOD.py...")
with open("causallearn/search/ConstraintBased/CDNOD.py", "r") as f:
    content = f.read()
    if 'depth_limit = kwargs.get("depth_limit", None)' in content:
        print("   ✅ depth_limit extracted from kwargs")
    else:
        print("   ❌ depth_limit NOT extracted from kwargs")

# Check 5: adaptive depth_limit in FedCDH
print("\n5. Checking adaptive depth_limit in FedCDH.py...")
with open("causallearn/search/FCMBased/FedCDH/FedCDH.py", "r") as f:
    content = f.read()
    if "default_depth_limit = min(4, max(2, int(np.log(n_samples) / 2)))" in content:
        print("   ✅ Adaptive depth_limit calculation")
    else:
        print("   ❌ Adaptive depth_limit NOT found")

    if '"depth_limit": depth_limit' in content:
        print("   ✅ depth_limit passed to cdnod_kwargs")
    else:
        print("   ❌ depth_limit NOT passed to cdnod_kwargs")

print()
print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
