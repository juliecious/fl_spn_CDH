import sys
import torch
import numpy as np
import logging
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from causallearn.utils.FedPC import LocalSPNWrapper, GlobalFedSPN
from causallearn.utils.cit import SPN_CIT

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def test_spn_ci_triangle():
    logging.info(">>> Starting SPN CI 'Triangle' Test (X -> Z -> Y)")

    # 1. Data Generation: X -> Z -> Y
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 2000

    # X ~ N(0, 1)
    X = np.random.normal(0, 1, n_samples)
    # Z = X + N(0, 0.5)
    Z = X + np.random.normal(0, 0.5, n_samples)
    # Y = Z + N(0, 0.5)
    Y = Z + np.random.normal(0, 0.5, n_samples)

    # Data Matrix: [X, Y, Z]
    # Indices: X=0, Y=1, Z=2
    data = np.stack([X, Y, Z], axis=1).astype(np.float32)

    logging.info(f"Generated {n_samples} samples.")
    logging.info("Training Federated SPN on [X, Y, Z]...")

    # 2. Train SPN
    # For simplicity, treat as 1 Client (Horizontal FL with K=1)
    # Using sufficient depth/leaves to capture dependencies
    spn = LocalSPNWrapper(
        num_features=3,
        device="cpu",
        num_sums=10,
        num_leaves=10,
        depth=1,  # log2(3) ~ 1.58 -> 1
        num_repetitions=5,
    )

    loss = spn.train_local(data, epochs=50, lr=0.01)
    logging.info(f"SPN Trained. Loss: {loss:.4f}")

    global_spn = GlobalFedSPN([spn], weights=[1.0])

    # 3. Initialize CI Oracle
    # Threshold is somewhat arbitrary for raw scores, but usually < 0.05 is indep
    # We will look at the raw scores first.
    oracle = SPN_CIT(data, global_model=global_spn, threshold=0.02)

    # 4. Test 1: Marginal Independence X _|_ Y ?
    # Ground Truth: False (Dependent)
    # Score should be HIGH
    # Pass 'data_matrix' explicitly as required by our updated signature logic if needed
    # but SPN_CIT stores self.data.

    # Calculate Raw Score first to inspect
    # Score = LL(XY) - LL(X) - LL(Y) = PMI (approx MI)
    # Wait, the formula in SPN_CIT is:
    # Score = LL(XYZ) - (LL(XZ) + LL(YZ) - LL(Z)) for conditional
    # For Marginal (Z=[]):
    # Score = LL(XY) - (LL(X) + LL(Y))

    logging.info(
        "Testing Marginal Independence: X _|_ Y (Expect Dependence/High Score)..."
    )

    # Manually compute score to log it
    # We use the internal logic of SPN_CIT
    # X=0, Y=1, Z=[]

    def get_score(x, y, z):
        # Access internal helper if possible, or just re-implement for checking
        # But let's use the public call and check the binary result first,
        # and maybe add a method to get the score if we need to debug.
        # Actually, let's just inspect the result.

        # We can also compute manually using the model
        if z:
            ll_xyz = oracle._get_marginal_log_prob(data, x + y + z)
            ll_xz = oracle._get_marginal_log_prob(data, x + z)
            ll_yz = oracle._get_marginal_log_prob(data, y + z)
            ll_z = oracle._get_marginal_log_prob(data, z)
            score = ll_xyz - (ll_xz + ll_yz - ll_z)
        else:
            ll_xy = oracle._get_marginal_log_prob(data, x + y)
            ll_x = oracle._get_marginal_log_prob(data, x)
            ll_y = oracle._get_marginal_log_prob(data, y)
            score = ll_xy - (ll_x + ll_y)
        return max(0.0, score)

    score_marginal = get_score([0], [1], [])
    is_indep_marginal = oracle(0, 1, [])

    logging.info(f"   X _|_ Y Score: {score_marginal:.4f}")
    logging.info(f"   Result: {'Independent' if is_indep_marginal else 'Dependent'}")

    if score_marginal < 0.05:
        logging.error("FAILURE: X and Y should be dependent (Score too low).")
        sys.exit(1)

    # 5. Test 2: Conditional Independence X _|_ Y | Z ?
    # Ground Truth: True (Independent)
    # Score should be LOW

    logging.info(
        "Testing Conditional Independence: X _|_ Y | Z (Expect Independence/Low Score)..."
    )

    score_cond = get_score([0], [1], [2])
    is_indep_cond = oracle(0, 1, [2])

    logging.info(f"   X _|_ Y | Z Score: {score_cond:.4f}")
    logging.info(f"   Result: {'Independent' if is_indep_cond else 'Dependent'}")

    if score_cond > 0.1:  # Allow some margin for estimation noise
        logging.error(
            f"FAILURE: X and Y should be independent given Z (Score {score_cond:.4f} too high)."
        )
        sys.exit(1)

    logging.info("SUCCESS: SPN Oracle correctly identifies dependence patterns.")


if __name__ == "__main__":
    test_spn_ci_triangle()
