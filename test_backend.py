# test_backend.py   
# run as: pytest -q test_backend.py
import io
import math
import numpy as np
import pytest

import backend

# -------------------------
# Sample data 
# -------------------------
COL1 = [5.30, 2.52, 3.68, 6.14, 7.29, 5.11, 4.92, 6.46, 2.25, 8.15]  # 10 entries (used for one-sample and two-sample tests)
COL2 = [5.76, 6.78, 4.98, 3.33, 5.58, 5.03, 3.53, 5.07, 6.68, 6.72, 5.93, 3.46]  # 12 entries (used for two-sample)

# Helper for deciding equivalence in tests
def tost_equivalent(p_low, p_high, alpha=0.05):
    # Equivalence declared when both one-sided p-values are < alpha (i.e. reject non-equivalence).
    return max(p_low, p_high) < alpha


# -------------------------
# Tests for one-sample equivalence test.
# -------------------------
def test_1_one_sample_tost_sample_data_minus_half_to_half():
    # One-sample TOST for the provided one-sample data against mu0=5.
    # Bounds (-0.5, 0.5) -> expected p_low ~= 0.147, p_high ~= 0.308
    
    p_low, p_high, _ = backend.one_sample_tost(COL1, mu0=5.0, lower=-0.5, upper=0.5, alpha=0.05)
    assert p_low == pytest.approx(0.147, rel=1e-2)
    assert p_high == pytest.approx(0.308, rel=1e-2)


def test_2_one_sample_tost_sample_data_minus_one_to_one():
    # One-sample TOST for the provided one-sample data against mu0=5.
    # Bounds (-1.0, 1.0) -> expected p_low ~= 0.043, p_high ~= 0.107
    
    p_low2, p_high2, _ = backend.one_sample_tost(COL1, mu0=5.0, lower=-1.0, upper=1.0, alpha=0.05)
    assert p_low2 == pytest.approx(0.043, rel=1e-2)
    assert p_high2 == pytest.approx(0.107, rel=1e-2)

def test_3_one_sample_generated_equivalent():
    # One-sample generated data that should be equivalent.
    
    rng = np.random.default_rng(1234)

    # Equivalent: mean ~ 0.05, small sd, n large -> within [-0.5, 0.5]
    equiv_data = rng.normal(loc=0.05, scale=0.15, size=100)
    p_low_e, p_high_e, _ = backend.one_sample_tost(equiv_data, mu0=0.0, lower=-0.5, upper=0.5, alpha=0.05)
    assert tost_equivalent(p_low_e, p_high_e, alpha=0.05), "Expected equivalence for equiv_data"


def test_4_one_sample_generated_non_equivalent():
    # One-sample generated data that should NOT be equivalent.
    
    rng = np.random.default_rng(1234)

    # Non-equivalent: mean ~ 1.0, larger difference -> outside [-0.5,0.5]
    non_equiv_data = rng.normal(loc=1.0, scale=0.3, size=100)
    p_low_ne, p_high_ne, _ = backend.one_sample_tost(non_equiv_data, mu0=0.0, lower=-0.5, upper=0.5, alpha=0.05)
    assert not tost_equivalent(p_low_ne, p_high_ne, alpha=0.05), "Expected non-equivalence for non_equiv_data"


# -------------------------
# Pooled t-test tests
# -------------------------
def test_5_two_sample_pooled_sample_data():
    # Two-sample pooled-variance TOST on the flipped columns (COL1, COL2).
    # Expected approximate p-values similar to those reported previously:
    #  pooled: p_low ~= 0.262, p_high ~= 0.214
    
    p_low, p_high, _ = backend.two_sample_pooled_tost(COL1, COL2, lower=-0.5, upper=0.5, alpha=0.05)
    assert p_low == pytest.approx(0.262, rel=1e-2)
    assert p_high == pytest.approx(0.214, rel=1e-2)


def test_6_two_sample_pooled_generated_equivalent():
    # Two-sample pooled test with generated data expected to be equivalent.
    
    rng = np.random.default_rng(27)

    # Equivalent pair: close means, same variance
    a_eq = rng.normal(loc=0.05, scale=0.5, size=100)
    b_eq = rng.normal(loc=-0.02, scale=0.5, size=100)
    p_low_eq, p_high_eq, _ = backend.two_sample_pooled_tost(a_eq, b_eq, lower=-0.5, upper=0.5, alpha=0.05)
    assert tost_equivalent(p_low_eq, p_high_eq, alpha=0.05)


def test_7_two_sample_pooled_generated_non_equivalent():
    # Two-sample pooled test with generated data expected to not be equivalent.
  
    rng = np.random.default_rng(16)

    # Non-equivalent pair: means differ markedly
    a_ne = rng.normal(loc=1.5, scale=0.8, size=100)
    b_ne = rng.normal(loc=-0.2, scale=0.8, size=100)
    p_low_ne, p_high_ne, _ = backend.two_sample_pooled_tost(a_ne, b_ne, lower=-0.5, upper=0.5, alpha=0.05)
    assert not tost_equivalent(p_low_ne, p_high_ne, alpha=0.05)


def test_8_two_sample_pooled_wrapper_vs_summary_core_consistency():
    # Pooled wrapper vs summary base function should agree for the same data.
    
    rng = np.random.default_rng(49)
    a_eq = rng.normal(loc=0.05, scale=0.6, size=100)
    b_eq = rng.normal(loc=-0.02, scale=0.6, size=100)
    p_low_eq, p_high_eq, _ = backend.two_sample_pooled_tost(a_eq, b_eq, lower=-0.5, upper=0.5, alpha=0.05)

    m1, s1, n1 = backend.summarize(a_eq)
    m2, s2, n2 = backend.summarize(b_eq)
    p_low_core, p_high_core, _ = backend.two_sample_pooled_tost_from_summary(m1, s1, n1, m2, s2, n2, lower=-0.5, upper=0.5, alpha=0.05)
    assert p_low_core == pytest.approx(p_low_eq, rel=1e-9)
    assert p_high_core == pytest.approx(p_high_eq, rel=1e-9)


# -------------------------
# Welch t-test tests
# -------------------------
def test_9_two_sample_welch_sample_data():
    # Welch two-sample TOST on the flipped columns. Expect:
    #  p_low ~= 0.271, p_high ~= 0.224
    
    p_low, p_high, _ = backend.two_sample_welch_tost(COL1, COL2, lower=-0.5, upper=0.5, alpha=0.05)
    assert p_low == pytest.approx(0.271, rel=1e-2)
    assert p_high == pytest.approx(0.224, rel=1e-2)

def test_10_two_sample_welch_generated_equivalent():
    # Welch two-sample test with generated data expected to be equivalent.
    
    rng = np.random.default_rng(4)

    # Equivalent under Welch: similar means, different variance
    a_eq = rng.normal(loc=0.02, scale=0.5, size=100)
    b_eq = rng.normal(loc=-0.04, scale=1.2, size=100)
    p_low_eq, p_high_eq, _ = backend.two_sample_welch_tost(a_eq, b_eq, lower=-0.5, upper=0.5, alpha=0.05)
    assert tost_equivalent(p_low_eq, p_high_eq, alpha=0.05)


def test_11_two_sample_welch_generated_non_equivalent():
    # Welch two-sample test with generated data expected to not be equivalent.
    
    rng = np.random.default_rng(9)

    # Non-equivalent: means far apart
    a_ne = rng.normal(loc=2.0, scale=0.6, size=100)
    b_ne = rng.normal(loc=-0.5, scale=1.0, size=100)
    p_low_ne, p_high_ne, _ = backend.two_sample_welch_tost(a_ne, b_ne, lower=-0.5, upper=0.5, alpha=0.05)
    assert not tost_equivalent(p_low_ne, p_high_ne, alpha=0.05)


def test_12_two_sample_welch_wrapper_vs_summary_core_consistency():
    # Welch wrapper vs summary base function should agree for the same data.
    
    rng = np.random.default_rng(64)
    a_eq = rng.normal(loc=0.02, scale=0.5, size=60)
    b_eq = rng.normal(loc=-0.04, scale=1.2, size=70)
    p_low_eq, p_high_eq, _ = backend.two_sample_welch_tost(a_eq, b_eq, lower=-0.5, upper=0.5, alpha=0.05)

    m1, s1, n1 = backend.summarize(a_eq)
    m2, s2, n2 = backend.summarize(b_eq)
    p_low_core, p_high_core, _ = backend.two_sample_welch_tost_from_summary(m1, s1, n1, m2, s2, n2, lower=-0.5, upper=0.5, alpha=0.05)
    assert p_low_core == pytest.approx(p_low_eq, rel=1e-9)
    assert p_high_core == pytest.approx(p_high_eq, rel=1e-9)


# -------------------------
# Brunner-Munzel (independent) tests
# -------------------------
def test_brunner_munzel_independent_sample_data():
    """Brunner-Munzel on COL1 vs COL2 with bounds [0.35, 0.65]."""
    p_low, p_high, details = backend.brunner_munzel_tost(COL1, COL2, lower=0.35, upper=0.65, alpha=0.05)
    # Validate theta_hat is between 0 and 1
    assert 0 < details["theta_hat"] < 1
    # Validate df > 0
    assert details["df"] > 0
    # Validate CI is within [0, 1]
    assert 0 <= details["ci_lower"] <= details["ci_upper"] <= 1
    # Validate p-values are valid probabilities
    assert 0 <= p_low <= 1
    assert 0 <= p_high <= 1


def test_brunner_munzel_independent_equivalent():
    """Generated data expected to be equivalent (close distributions)."""
    rng = np.random.default_rng(42)
    a = rng.normal(loc=0.0, scale=1.0, size=100)
    b = rng.normal(loc=0.0, scale=1.0, size=100)
    p_low, p_high, details = backend.brunner_munzel_tost(a, b, lower=0.35, upper=0.65, alpha=0.05)
    assert tost_equivalent(p_low, p_high, alpha=0.05), "Expected equivalence for similar distributions"


def test_brunner_munzel_independent_non_equivalent():
    """Generated data expected NOT to be equivalent (separated distributions)."""
    rng = np.random.default_rng(42)
    a = rng.normal(loc=0.0, scale=1.0, size=100)
    b = rng.normal(loc=3.0, scale=1.0, size=100)
    p_low, p_high, details = backend.brunner_munzel_tost(a, b, lower=0.35, upper=0.65, alpha=0.05)
    assert not tost_equivalent(p_low, p_high, alpha=0.05), "Expected non-equivalence for separated distributions"


def test_brunner_munzel_independent_small_sample_warning():
    """Small sample should trigger warning."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=10)
    b = rng.normal(size=10)
    _, _, details = backend.brunner_munzel_tost(a, b, lower=0.35, upper=0.65, alpha=0.05)
    assert len(details["warnings"]) > 0, "Expected small-sample warning"


# -------------------------
# Paired Brunner-Munzel tests
# -------------------------
def test_paired_brunner_munzel_sample_data():
    """Paired BM on COL1[:10] vs COL2[:10] (first 10 elements, truncated to equal length)."""
    x = np.array(COL1[:10])
    y = np.array(COL2[:10])
    p_low, p_high, details = backend.paired_brunner_munzel_tost(x, y, lower=0.35, upper=0.65, alpha=0.05)
    # Validate structure
    assert 0 < details["theta_hat"] < 1
    assert details["df"] == len(x) - 1  # df = n-1 for paired
    assert 0 <= details["ci_lower"] <= details["ci_upper"] <= 1
    assert 0 <= p_low <= 1
    assert 0 <= p_high <= 1


def test_paired_brunner_munzel_equivalent():
    """Paired data with small shift should be equivalent with wide bounds."""
    rng = np.random.default_rng(55)
    baseline = rng.normal(loc=5.0, scale=1.0, size=80)
    treatment = baseline + rng.normal(loc=0.0, scale=0.3, size=80)  # Small shift
    p_low, p_high, details = backend.paired_brunner_munzel_tost(
        baseline, treatment, lower=0.35, upper=0.65, alpha=0.05
    )
    assert tost_equivalent(p_low, p_high, alpha=0.05), "Expected equivalence for small paired shift"


def test_paired_brunner_munzel_non_equivalent():
    """Paired data with large shift should NOT be equivalent."""
    rng = np.random.default_rng(55)
    baseline = rng.normal(loc=5.0, scale=1.0, size=80)
    treatment = baseline + 3.0  # Large systematic shift
    p_low, p_high, details = backend.paired_brunner_munzel_tost(
        baseline, treatment, lower=0.35, upper=0.65, alpha=0.05
    )
    assert not tost_equivalent(p_low, p_high, alpha=0.05), "Expected non-equivalence for large shift"


def test_paired_brunner_munzel_df_is_n_minus_1():
    """Verify that df = n - 1 (not Satterthwaite)."""
    rng = np.random.default_rng(99)
    n = 25
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    _, _, details = backend.paired_brunner_munzel_tost(x, y, lower=0.35, upper=0.65, alpha=0.05)
    assert details["df"] == n - 1


def test_paired_brunner_munzel_requires_equal_length():
    """Unequal-length inputs should raise an error."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0])
    with pytest.raises((ValueError, Exception)):
        backend.paired_brunner_munzel_tost(x, y, lower=0.35, upper=0.65, alpha=0.05)


def test_paired_brunner_munzel_small_sample_warning():
    """Small sample should trigger warning."""
    rng = np.random.default_rng(7)
    n = 15
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    _, _, details = backend.paired_brunner_munzel_tost(x, y, lower=0.35, upper=0.65, alpha=0.05)
    assert len(details["warnings"]) > 0, "Expected small-sample warning for n < 30"


# -------------------------
# Proportions tests
# -------------------------
def test_proportions_difference_equivalent():
    """Large-sample proportions close together should be equivalent."""
    # p1 = 100/1000 = 0.10, p2 = 95/1000 = 0.095
    p_low, p_high, details = backend.proportions_tost_difference(100, 1000, 95, 1000, lower=-0.05, upper=0.05, alpha=0.05)
    assert tost_equivalent(p_low, p_high, alpha=0.05), "Expected equivalence for close proportions"


def test_proportions_difference_non_equivalent():
    """Proportions far apart should NOT be equivalent with tight bounds."""
    # p1 = 100/1000 = 0.10, p2 = 300/1000 = 0.30
    p_low, p_high, details = backend.proportions_tost_difference(100, 1000, 300, 1000, lower=-0.05, upper=0.05, alpha=0.05)
    assert not tost_equivalent(p_low, p_high, alpha=0.05), "Expected non-equivalence for disparate proportions"


def test_proportions_odds_ratio():
    """Basic OR TOST should return valid p-values and details."""
    p_low, p_high, details = backend.proportions_tost_odds_ratio(50, 200, 55, 200, lower=0.8, upper=1.25, alpha=0.05)
    assert 0 <= p_low <= 1
    assert 0 <= p_high <= 1
    assert "estimate" in details


def test_proportions_risk_ratio():
    """Basic RR TOST should return valid p-values and details."""
    p_low, p_high, details = backend.proportions_tost_risk_ratio(50, 200, 55, 200, lower=0.8, upper=1.25, alpha=0.05)
    assert 0 <= p_low <= 1
    assert 0 <= p_high <= 1


def test_proportions_small_sample_warning():
    """Small cell counts should trigger warnings."""
    _, _, details = backend.proportions_tost_difference(3, 20, 4, 20, lower=-0.1, upper=0.1, alpha=0.05)
    assert len(details.get("warnings", [])) > 0, "Expected warning for small sample"


def test_proportions_wald_test_difference():
    """Wald z-test for two-sided difference should return valid p-value."""
    p = backend.proportions_wald_test_difference(50, 200, 55, 200, pooled=False, alternative="two-sided")
    assert 0 <= p <= 1


# ===========================================================================
# R-VALIDATED TESTS
# These expected values were computed in R using TOSTER package.
# To regenerate, run the corresponding R code in the comment above each test.
# ===========================================================================

def test_brunner_munzel_independent_vs_R():
    """
    Validate against R:
        library(TOSTER)
        x <- c(5.30, 2.52, 3.68, 6.14, 7.29, 5.11, 4.92, 6.46, 2.25, 8.15)
        y <- c(5.76, 6.78, 4.98, 3.33, 5.58, 5.03, 3.53, 5.07, 6.68, 6.72, 5.93, 3.46)
        res <- brunner_munzel(x, y, alternative = "equivalence", mu = c(0.35, 0.65))
        # Extract: res$estimate, res$p.value, res$conf.int, res$parameter
    """
    x = np.array(COL1)
    y = np.array(COL2)
    p_low, p_high, details = backend.brunner_munzel_tost(x, y, lower=0.35, upper=0.65, alpha=0.05)
    # Fill in expected values from R after running the above code.
    # assert details["theta_hat"] == pytest.approx(EXPECTED_THETA, rel=1e-3)
    # assert max(p_low, p_high) == pytest.approx(EXPECTED_TOST_P, rel=1e-2)
    pass  # TODO: Fill with R output


def test_paired_brunner_munzel_vs_R():
    """
    Validate against R:
        library(TOSTER)
        x <- c(5.30, 2.52, 3.68, 6.14, 7.29, 5.11, 4.92, 6.46, 2.25, 8.15)
        y <- c(5.76, 6.78, 4.98, 3.33, 5.58, 5.03, 3.53, 5.07, 6.68, 6.72)
        res <- brunner_munzel(x, y, paired = TRUE, alternative = "equivalence", mu = c(0.35, 0.65))
        # Extract: res$estimate, res$p.value, res$conf.int, res$parameter
    """
    x = np.array(COL1[:10])
    y = np.array(COL2[:10])
    p_low, p_high, details = backend.paired_brunner_munzel_tost(x, y, lower=0.35, upper=0.65, alpha=0.05)
    # Fill in expected values from R after running the above code.
    # assert details["theta_hat"] == pytest.approx(EXPECTED_THETA, rel=1e-3)
    # assert details["df"] == 9  # n - 1
    # assert max(p_low, p_high) == pytest.approx(EXPECTED_TOST_P, rel=1e-2)
    pass  # TODO: Fill with R output


if __name__ == "__main__":
    pytest.main([__file__])
