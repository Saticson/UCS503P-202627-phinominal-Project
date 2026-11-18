"""
Sukio / AxiomCloud - Mathematical Correctness Unit Tests for Conformal Prediction
Validates:
1. Finite-sample quantile formula: ceil((n + 1) * (1 - alpha)) / n
2. Marginal empirical coverage under true exchangeability (guarantee >= 1 - alpha)
3. Class-conditional (Mondrian) empirical coverage under exchangeability
4. Nonconformity score orientation (1 - P(Y=y)) and empty prediction set handling
5. Mathematical demonstration of coverage loss when exchangeability is violated (distribution shift)
"""

import math
import numpy as np
import pytest

def compute_conformal_quantile(scores, alpha=0.10):
    n = len(scores)
    if n == 0:
        return 1.0
    sorted_scores = np.sort(scores)
    # Exact finite-sample quantile index: ceil((n+1)(1-alpha)) - 1 (0-indexed)
    q_idx = int(math.ceil((n + 1) * (1.0 - alpha))) - 1
    q_idx = max(0, min(n - 1, q_idx))
    return float(sorted_scores[q_idx])

def compute_prediction_set(probs, quantiles, allow_empty=False):
    """
    probs: dict {class_idx: probability}
    quantiles: dict {class_idx: q_c} or float for marginal
    """
    prediction_set = []
    for c, p in probs.items():
        q_thresh = quantiles[c] if isinstance(quantiles, dict) else quantiles
        # In split-conformal, nonconformity is (1 - p). In set if 1 - p <= q => p >= 1 - q
        if (1.0 - p) <= q_thresh:
            prediction_set.append(c)
    if not prediction_set and not allow_empty:
        # Prevent empty set by returning top-1 argmax
        prediction_set = [max(probs.items(), key=lambda kv: kv[1])[0]]
    return prediction_set

# ----------------------------------------------------------------------
# 1. Finite-Sample Quantile Index Known-Answer Tests
# ----------------------------------------------------------------------
def test_finite_sample_quantile_known_answer():
    # n = 9, alpha = 0.10: (9+1)*0.9 = 10 -> index 9 (the 10th value, but n=9 so clamped to max)
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    q = compute_conformal_quantile(scores, alpha=0.10)
    assert q == 0.9, f"Expected 0.9, got {q}"

    # n = 19, alpha = 0.10: ceil(20 * 0.9) - 1 = 18 - 1 = 17 (0-indexed 17th element)
    scores_20 = np.linspace(0.05, 1.0, 19)
    q_20 = compute_conformal_quantile(scores_20, alpha=0.10)
    assert q_20 == scores_20[17]

# ----------------------------------------------------------------------
# 2. Marginal Coverage Under Exchangeability (Monte Carlo Analytical Proof)
# ----------------------------------------------------------------------
def test_marginal_coverage_under_exchangeability():
    np.random.seed(42)
    alpha = 0.10
    n_cal = 400
    n_test = 400
    n_trials = 200

    trial_coverages = []
    for _ in range(n_trials):
        # Exchangeable nonconformity scores drawn from identical Beta distribution
        cal_scores = np.random.beta(1.0, 4.0, size=n_cal)
        test_scores = np.random.beta(1.0, 4.0, size=n_test)

        q = compute_conformal_quantile(cal_scores, alpha=alpha)
        coverage = np.mean(test_scores <= q)
        trial_coverages.append(coverage)

    mean_cov = np.mean(trial_coverages)
    # Theoretical expectation is exactly 1 - alpha = 0.90 (>= 0.90 by finite-sample conservative bound)
    assert mean_cov >= (1.0 - alpha) - 0.005, f"Expected coverage >= 0.895, got {mean_cov}"

# ----------------------------------------------------------------------
# 3. Class-Conditional (Mondrian) Coverage Under Exchangeability
# ----------------------------------------------------------------------
def test_mondrian_coverage_under_exchangeability():
    np.random.seed(42)
    alpha = 0.10
    n_cal_per_class = 200
    n_test_per_class = 200

    # Class 0 has higher confidence than Class 1
    cal_s0 = np.random.beta(1.0, 8.0, size=n_cal_per_class)
    test_s0 = np.random.beta(1.0, 8.0, size=n_test_per_class)

    cal_s1 = np.random.beta(2.0, 4.0, size=n_cal_per_class)
    test_s1 = np.random.beta(2.0, 4.0, size=n_test_per_class)

    q0 = compute_conformal_quantile(cal_s0, alpha=alpha)
    q1 = compute_conformal_quantile(cal_s1, alpha=alpha)

    cov0 = np.mean(test_s0 <= q0)
    cov1 = np.mean(test_s1 <= q1)

    assert cov0 >= (1.0 - alpha) - 0.02, f"Class 0 coverage failed: {cov0}"
    assert cov1 >= (1.0 - alpha) - 0.02, f"Class 1 coverage failed: {cov1}"

# ----------------------------------------------------------------------
# 4. Exchangeability Violation Demonstration (PneumoniaMNIST Analogue)
# ----------------------------------------------------------------------
def test_coverage_collapse_under_distribution_shift():
    """
    Demonstrates mathematically why PneumoniaMNIST achieved ~79% coverage:
    Calibration scores were overconfident (mean ~0.009, q ~0.012),
    while Test scores experienced covariate shift (mean ~0.138).
    Conformal guarantee mathematically requires exchangeability.
    """
    np.random.seed(42)
    # Calibration set: artificially easy / overconfident
    cal_scores = np.random.uniform(0.001, 0.015, size=200)
    q = compute_conformal_quantile(cal_scores, alpha=0.10)

    # Test set: out-of-distribution / harder cohort
    test_scores = np.random.uniform(0.010, 0.200, size=200)
    shifted_cov = np.mean(test_scores <= q)

    # Coverage collapses because test distribution violates exchangeability
    assert shifted_cov < 0.70, f"Expected coverage collapse under shift, got {shifted_cov}"

# ----------------------------------------------------------------------
# 5. Nonconformity Score Orientation & Empty Set Prevention
# ----------------------------------------------------------------------
def test_empty_set_prevention():
    probs = {0: 0.49, 1: 0.51}
    # If quantiles are overly strict (e.g. 0.10 => threshold p >= 0.90)
    quantiles = {0: 0.10, 1: 0.10}

    # Empty set allowed: returns empty list
    empty_res = compute_prediction_set(probs, quantiles, allow_empty=True)
    assert empty_res == []

    # Non-empty set enforcement: returns argmax (Class 1)
    safe_res = compute_prediction_set(probs, quantiles, allow_empty=False)
    assert safe_res == [1], f"Expected fallback argmax [1], got {safe_res}"

if __name__ == "__main__":
    test_finite_sample_quantile_known_answer()
    test_marginal_coverage_under_exchangeability()
    test_mondrian_coverage_under_exchangeability()
    test_coverage_collapse_under_distribution_shift()
    test_empty_set_prevention()
    print("ALL CONFORMAL MATHEMATICAL CORRECTNESS TESTS PASSED.")
