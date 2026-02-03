import sys
import torch
import numpy as np
import logging
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from causallearn.utils.FedPC import LocalSPNWrapper, GlobalFedSPN

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def test_fed_spn_mixture():
    logging.info(">>> Starting FedSPN Mixture 'Castle' Test")

    # 1. Data Generation
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 1000
    d_features = 2

    # Client A: N(-5, 1) in 2D
    data_A = np.random.normal(loc=-5.0, scale=1.0, size=(n_samples, d_features)).astype(
        np.float32
    )
    # Client B: N(5, 1) in 2D
    data_B = np.random.normal(loc=5.0, scale=1.0, size=(n_samples, d_features)).astype(
        np.float32
    )

    logging.info(f"Generated Data A: Mean={data_A.mean():.2f}, Std={data_A.std():.2f}")
    logging.info(f"Generated Data B: Mean={data_B.mean():.2f}, Std={data_B.std():.2f}")

    # 2. Train Local Models
    logging.info("Initializing and Training LocalSPN A...")
    spn_a = LocalSPNWrapper(
        num_features=d_features, num_sums=5, num_leaves=5, depth=1, num_repetitions=1
    )
    loss_a = spn_a.train_local(data_A, epochs=50, lr=0.05)
    logging.info(f"LocalSPN A trained. Loss: {loss_a:.4f}")

    logging.info("Initializing and Training LocalSPN B...")
    spn_b = LocalSPNWrapper(
        num_features=d_features, num_sums=5, num_leaves=5, depth=1, num_repetitions=1
    )
    loss_b = spn_b.train_local(data_B, epochs=50, lr=0.05)
    logging.info(f"LocalSPN B trained. Loss: {loss_b:.4f}")

    # 3. Construct Global SPN (The Castle)
    # Weights assumed uniform: 0.5, 0.5
    global_spn = GlobalFedSPN([spn_a, spn_b], weights=[0.5, 0.5])

    # 4. Query Test Point x = [5.0, 5.0]
    # Expected:
    # P_A(5) should be effectively 0 (10 std devs away)
    # P_B(5) should be high (mean of distribution)
    # P_Global(5) should be approx 0.5 * P_B(5)

    test_x = torch.tensor([[5.0, 5.0]], dtype=torch.float32)

    with torch.no_grad():
        ll_a = spn_a.log_prob(test_x).item()
        ll_b = spn_b.log_prob(test_x).item()
        ll_global = global_spn.log_prob(test_x).item()

        # New: Conditional Queries
        ll_cond_a = global_spn.log_prob_conditional_u(test_x, 0).item()
        ll_cond_b = global_spn.log_prob_conditional_u(test_x, 1).item()

    prob_a = np.exp(ll_a)
    prob_b = np.exp(ll_b)
    prob_global = np.exp(ll_global)
    prob_cond_a = np.exp(ll_cond_a)
    prob_cond_b = np.exp(ll_cond_b)

    logging.info("-" * 40)
    logging.info(f"Test Point x = [5.0, 5.0]")
    logging.info(
        f"Local Model A (trained on -5): LogProb={ll_a:.4f}, Prob={prob_a:.6f}"
    )
    logging.info(
        f"Local Model B (trained on +5): LogProb={ll_b:.4f}, Prob={prob_b:.6f}"
    )
    logging.info(
        f"Global Model (Mixture):        LogProb={ll_global:.4f}, Prob={prob_global:.6f}"
    )
    logging.info(
        f"Global Cond (U=0/A):           LogProb={ll_cond_a:.4f}, Prob={prob_cond_a:.6f}"
    )
    logging.info(
        f"Global Cond (U=1/B):           LogProb={ll_cond_b:.4f}, Prob={prob_cond_b:.6f}"
    )
    logging.info("-" * 40)

    # Verification Logic
    # 1. Prob A should be tiny
    if prob_a > 1e-3:
        logging.error("FAILURE: Local Model A probability is too high for x=5.")
        sys.exit(1)

    # 2. Prob B should be reasonable (Product of 2 Normals at mean)
    # PDF(0) ~ 0.4. PDF(0,0) ~ 0.16.
    if prob_b < 0.05:
        logging.error(
            f"FAILURE: Local Model B probability is too low for x=5 (Got {prob_b})."
        )
        sys.exit(1)

    # 3. Global Prob should be approx average
    expected_global = 0.5 * prob_a + 0.5 * prob_b
    if not np.isclose(prob_global, expected_global, rtol=1e-3):
        logging.error(
            f"FAILURE: Global Probability mismatch. Expected ~{expected_global}, Got {prob_global}"
        )
        sys.exit(1)

    # 4. Conditional Verification
    if not np.isclose(prob_cond_a, prob_a):
        logging.error("FAILURE: Conditional U=0 should match Local A")
        sys.exit(1)
    if not np.isclose(prob_cond_b, prob_b):
        logging.error("FAILURE: Conditional U=1 should match Local B")
        sys.exit(1)

    logging.info(
        "SUCCESS: The Global SPN correctly supports conditional queries P(X|U)."
    )

    logging.info(
        "SUCCESS: The Global SPN correctly aggregates local experts without corruption."
    )


if __name__ == "__main__":
    test_fed_spn_mixture()
