# helpers.py — formatting & plain-language summary helpers
"""
Utility functions that support the Streamlit UI layer (app.py).

- `update_parsed_counts`: parse a DataFrame into a summary dict for display
- `describe_tost`: produce a plain-language summary paragraph of TOST results
"""

from typing import Optional
import numpy as np

# try to import scipy for extra numeric helpers used in CI computation
try:
    from scipy import stats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def update_parsed_counts(df_parsed, test_choice: str):
    """Return parsed_counts dict or None depending on test type and parsed df."""
    import pandas as pd
    if df_parsed is None:
        return None

    TWO_SAMPLE_TESTS = {
        "Welch t-test (two-sample)",
        "Pooled t-test (two-sample)",
        "Brunner-Munzel test",
        "Paired Brunner-Munzel test",  # Needs two columns too
        # "Hodges-Lehmann (two-sample)",  # Hidden
    }

    ONE_SAMPLE_TESTS = {
        "One-sample t-test",
        # "Paired concordance test",    # Hidden
        # "Hodges-Lehmann (paired)",    # Hidden
    }
    # Note: Hodges-Lehmann and Paired concordance branches retained for future use.

    try:
        if test_choice in TWO_SAMPLE_TESTS:
            n1 = pd.to_numeric(df_parsed.iloc[:, 0], errors="coerce").dropna().shape[0] if df_parsed.shape[1] >= 1 else 0
            n2 = pd.to_numeric(df_parsed.iloc[:, 1], errors="coerce").dropna().shape[0] if df_parsed.shape[1] >= 2 else 0
            return {"group1": int(n1), "group2": int(n2)}

        elif test_choice in ONE_SAMPLE_TESTS:
            n = pd.to_numeric(df_parsed.iloc[:, 0], errors="coerce").dropna().shape[0] if df_parsed.shape[1] >= 1 else 0
            return {"values": int(n)}

        elif test_choice == "Two-sample proportions":
            # Expect a 2x2 contingency table (2 rows, 2 cols of counts)
            if df_parsed.shape[0] >= 2 and df_parsed.shape[1] >= 2:
                a = pd.to_numeric(df_parsed.iloc[0, 0], errors="coerce")
                b = pd.to_numeric(df_parsed.iloc[0, 1], errors="coerce")
                c = pd.to_numeric(df_parsed.iloc[1, 0], errors="coerce")
                d = pd.to_numeric(df_parsed.iloc[1, 1], errors="coerce")
                if any(pd.isna(v) for v in [a, b, c, d]):
                    return None
                a = int(a); b = int(b); c = int(c); d = int(d)
                x1 = a; n1 = a + b
                x2 = c; n2 = c + d
                return {"x1": int(x1), "n1": int(n1), "x2": int(x2), "n2": int(n2)}
            return None
        else:
            return None
    except Exception:
        return None


def describe_tost(
    test_choice: str,
    p_low: Optional[float],
    p_high: Optional[float],
    p_diff: Optional[float],
    lower: float,
    upper: float,
    alpha: float,
    x: Optional[np.ndarray],
    y: Optional[np.ndarray],
    paired_a: Optional[np.ndarray],
    paired_b: Optional[np.ndarray],
    summ: Optional[dict],
    mu0: float,
    digits: int = 3,
) -> str:
    """
    Build a verbose description in the style of R's describe.TOSTt.
    """
    fmt_p = lambda v: ("< 0.001" if (v is not None and v < 0.001) else (f"{v:.{digits}g}" if v is not None else "NA"))
    p_tost = None
    try:
        if p_low is not None and p_high is not None:
            p_tost = max(p_low, p_high)
    except Exception:
        p_tost = None

    # Determine variable name and estimate label based on test choice
    if test_choice == "One-sample t-test":
        varname = "mean"
        est_label = "mean"
    elif test_choice in ("Welch t-test (two-sample)", "Pooled t-test (two-sample)"):
        varname = "mean difference"
        est_label = "mean difference"
    elif test_choice == "Brunner-Munzel test":
        varname = "relative effect (θ)"
        est_label = "relative effect"
    elif test_choice == "Paired Brunner-Munzel test":
        varname = "relative effect (θ)"
        est_label = "relative effect"
    elif test_choice == "Hodges-Lehmann (two-sample)":
        varname = "location shift"
        est_label = "Hodges-Lehmann estimate"
    elif test_choice == "Paired concordance test":
        varname = "concordance probability"
        est_label = "concordance probability"
    elif test_choice == "Hodges-Lehmann (paired)":
        varname = "pseudomedian"
        est_label = "pseudomedian"
    elif test_choice == "Two-sample proportions":
        varname = "effect size"
        est_label = "estimate"
    else:
        varname = "estimate"
        est_label = "estimate"

    method_state = (
        f"Using the {test_choice}, a null-hypothesis significance test (NHST) and an equivalence (TOST) test "
        f"were performed with alpha = {alpha}."
    )

    # Compute estimate + CI (best-effort)
    def compute_diff_and_ci():
        conf_level = max(0.0, 1.0 - 2.0 * alpha)
        try:
            if test_choice in ("Welch t-test (two-sample)", "Pooled t-test (two-sample)"):
                if x is not None and y is not None and len(x) > 0 and len(y) > 0:
                    m1 = float(np.nanmean(x))
                    m2 = float(np.nanmean(y))
                    diff = m1 - m2
                    sd1 = float(np.nanstd(x, ddof=1))
                    sd2 = float(np.nanstd(y, ddof=1))
                    n1 = x.size
                    n2 = y.size
                    if test_choice == "Pooled t-test (two-sample)":
                        sp2 = ((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2)
                        se = (sp2 * (1.0 / n1 + 1.0 / n2)) ** 0.5
                        df_ = n1 + n2 - 2
                    else:
                        se = (sd1 ** 2 / n1 + sd2 ** 2 / n2) ** 0.5
                        num = (sd1 ** 2 / n1 + sd2 ** 2 / n2) ** 2
                        den = ((sd1 ** 2 / n1) ** 2) / (n1 - 1) + ((sd2 ** 2 / n2) ** 2) / (n2 - 1)
                        df_ = (num / den) if den > 0 else None
                    if _HAS_SCIPY and se is not None and df_ and df_ > 0:
                        tail = (1.0 - conf_level) / 2.0
                        q = stats.t.ppf(1.0 - tail, df_)
                        ci_low = diff - q * se
                        ci_high = diff + q * se
                        return diff, ci_low, ci_high, int(conf_level * 100), f"mean1={m1:.{digits}g}, mean2={m2:.{digits}g}, n1={n1}, n2={n2}"
                    return diff, None, None, int(conf_level * 100), f"mean1={m1:.{digits}g}, mean2={m2:.{digits}g}, n1={n1}, n2={n2}"
                elif summ is not None and "m1" in summ:
                    diff = summ["m1"] - summ["m2"]
                    return diff, None, None, int(conf_level * 100), f"mean1={summ['m1']:.{digits}g}, mean2={summ['m2']:.{digits}g}"

            elif test_choice == "One-sample t-test":
                if x is not None and len(x) > 0:
                    m = float(np.nanmean(x))
                    n = x.size
                    diff = m - mu0
                    sd = float(np.nanstd(x, ddof=1))
                    se = sd / (n ** 0.5) if n > 0 else None
                    if _HAS_SCIPY and se is not None and n - 1 > 0:
                        tail = (1.0 - conf_level) / 2.0
                        q = stats.t.ppf(1.0 - tail, n - 1)
                        ci_low = diff - q * se
                        ci_high = diff + q * se
                        return diff, ci_low, ci_high, int(conf_level * 100), f"mean={m:.{digits}g}, n={n}"
                    return diff, None, None, int(conf_level * 100), f"mean={m:.{digits}g}, n={n}"
                elif summ is not None and "m" in summ:
                    diff = summ["m"] - mu0
                    return diff, None, None, int(conf_level * 100), f"mean={summ['m']:.{digits}g}, n={summ.get('n', '?')}"

            elif test_choice == "Brunner-Munzel test":
                # No raw estimate to compute here; app.py shows details
                return None, None, None, None, ""

            elif test_choice == "Paired Brunner-Munzel test":
                # Estimate and CI computed in backend; no raw recomputation needed here.
                return None, None, None, None, ""

            elif test_choice == "Hodges-Lehmann (two-sample)":
                if x is not None and y is not None:
                    diffs = np.subtract.outer(x, y).ravel()
                    hl = float(np.median(diffs))
                    return hl, None, None, None, f"n1={len(x)}, n2={len(y)}"
                return None, None, None, None, ""

            elif test_choice in ("Paired concordance test", "Hodges-Lehmann (paired)"):
                return None, None, None, None, ""

            elif test_choice == "Two-sample proportions":
                if summ and all(k in summ for k in ("x1", "n1", "x2", "n2")):
                    x1 = int(summ["x1"]); n1 = int(summ["n1"])
                    x2 = int(summ["x2"]); n2 = int(summ["n2"])
                    p1 = x1 / n1 if n1 > 0 else None
                    p2 = x2 / n2 if n2 > 0 else None
                    diff = (p1 - p2) if (p1 is not None and p2 is not None) else None
                    return diff, None, None, None, f"p1={p1:.{digits}g}, p2={p2:.{digits}g}"
                return None, None, None, None, ""

        except Exception:
            pass
        return None, None, None, None, ""

    diff, ci_low, ci_high, conf_pct, extra = compute_diff_and_ci()

    # Build stat_text and claim_text
    p_tost_text = fmt_p(p_tost) if p_tost is not None else "NA"
    p_nhst_text = fmt_p(p_diff) if p_diff is not None else "NA"

    if p_diff is not None and p_diff < alpha and (p_tost is None or p_tost >= alpha):
        stat_text = (
            f"The two-sided test for difference was significant (two-sided p = {p_nhst_text}). "
            f"The equivalence (TOST) test was not significant (p = {p_tost_text})."
        )
        claim_text = (
            f"At the desired error rate, it can be stated that the true {varname} differs from its null (i.e., not equivalent)."
        )
    elif p_tost is not None and p_tost < alpha:
        if p_diff is not None:
            diff_sig_text = "statistically significant" if p_diff < alpha else "not statistically significant"
            stat_text = (
                f"The two-sided test for difference has p = {p_nhst_text} — {diff_sig_text}. "
                f"The equivalence (TOST) test was significant (p = {p_tost_text})."
            )
        else:
            stat_text = f"The equivalence (TOST) test was significant (p = {p_tost_text}). The two-sided p-value for the difference is not available."
        claim_text = f"At the desired error rate, it can be stated that the {varname} is between {lower} and {upper}."
    else:
        stat_text = (
            f"Both the two-sided test for differences (two-sided p = {p_nhst_text}) and the equivalence (TOST) test (p = {p_tost_text}) were not significant."
        )
        claim_text = "Therefore, the results are inconclusive: neither null hypothesis can be rejected."

    # Compose estimate/CI sentence
    est_text = ""
    try:
        if diff is not None:
            est_text += f" (estimate = {diff:.{digits}g}"
            if ci_low is not None and ci_high is not None and conf_pct is not None:
                est_text += f"; {conf_pct}% C.I. [{ci_low:.{digits}g}, {ci_high:.{digits}g}]"
            if extra:
                est_text += f"; {extra}"
            est_text += ")"
    except Exception:
        est_text = ""

    full = f"{method_state} {stat_text} {claim_text}{est_text}"
    return full
