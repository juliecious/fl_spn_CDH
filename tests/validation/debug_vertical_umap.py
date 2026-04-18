"""
Debug script to check why vertical mode has no global UMAP.
"""
import sys
import os

# Find the most recent vertical eval directory
eval_dir = "eval"
import glob

vertical_dirs = sorted(
    glob.glob(f"{eval_dir}/*vertical*"), key=os.path.getmtime, reverse=True
)

if not vertical_dirs:
    print("No vertical eval directories found!")
    sys.exit(1)

latest_vertical = vertical_dirs[0]
print(f"Checking: {latest_vertical}\n")

# Check for UMAP files
import glob

umap_files = glob.glob(f"{latest_vertical}/umap_*.png")
print(f"UMAP files found: {len(umap_files)}")
for f in umap_files:
    print(f"  - {os.path.basename(f)}")

if "umap_global_spn.png" not in [os.path.basename(f) for f in umap_files]:
    print("\n❌ Global UMAP is MISSING\n")
else:
    print("\n✅ Global UMAP exists\n")

# Check run.log for clues
log_file = f"{latest_vertical}/run.log"
if os.path.exists(log_file):
    print("=" * 60)
    print("Checking run.log for issues:")
    print("=" * 60)

    with open(log_file, "r") as f:
        lines = f.readlines()

    # Look for relevant lines
    for i, line in enumerate(lines):
        if "dimension" in line.lower():
            print(f"Line {i}: {line.strip()}")
        if "global federated spn" in line.lower():
            # Print context around this line
            start = max(0, i - 2)
            end = min(len(lines), i + 10)
            print(
                f"\n--- Context around 'Global Federated SPN' (lines {start}-{end}) ---"
            )
            for j in range(start, end):
                print(f"{j}: {lines[j].rstrip()}")
            break

    # Check if d_features > 2
    for line in lines:
        if "Features (d):" in line:
            print(f"\n{line.strip()}")

    # Check for UMAP line
    found_umap_line = False
    for line in lines:
        if "umap_global_spn" in line.lower():
            print(f"\nFound UMAP mention: {line.strip()}")
            found_umap_line = True

    if not found_umap_line:
        print(
            "\n⚠️  No mention of 'umap_global_spn' in log - UMAP creation was skipped"
        )
        print("\nPossible reasons:")
        print("  1. d_features <= 2 (check Features (d) in log)")
        print("  2. Dimension mismatch (samples vs X_global)")
        print("  3. Sampling failed silently")
        print("  4. UMAP library not available")
else:
    print(f"❌ Log file not found: {log_file}")

print("\n" + "=" * 60)
print("Diagnostic complete")
print("=" * 60)
