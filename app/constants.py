"""
constants.py
------------
Single source of truth for values that were previously duplicated across
data_utils.py, model_utils.py, and app.py:
  - Payment / billing / repayment column names
  - The "is this row a new applicant" flag parser
  - Risk-tier thresholds, labels, and colors
  - The "was this decision an override of the model" rule
"""

from math import isfinite

# ---------------------------------------------------------------------------
# Column groups (previously re-typed in data_utils.py, model_utils.py, app.py)
# ---------------------------------------------------------------------------

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

# Human-readable labels for PAY_COLS, in the same order (current -> oldest).
PAY_COL_LABELS = ["Current", "2 Mo Ago", "3 Mo Ago", "4 Mo Ago", "5 Mo Ago", "6 Mo Ago"]


# ---------------------------------------------------------------------------
# "New applicant" flag parsing
# ---------------------------------------------------------------------------

_TRUTHY_STRINGS = {"true", "1", "1.0", "yes", "t"}


def is_new_applicant_flag(value) -> bool:
    """Interprets the various truthy formats the IS_NEW_APPLICANT column may
    contain (bool, int, or string) as a single boolean."""
    return str(value).strip().lower() in _TRUTHY_STRINGS


# ---------------------------------------------------------------------------
# Risk tiers
# ---------------------------------------------------------------------------

RISK_TIER_COLORS = {
    "LOW RISK": "#22c55e",
    "MODERATE RISK": "#eab308",
    "HIGH RISK": "#ef4444",
}


def classify_risk(
    default_prob: float,
    low_threshold: float = 40,
    high_threshold: float = 70
) -> tuple[str, str]:
    """
    Maps a default probability (0-100) to a (label, color) risk tier.

    Parameters
    ----------
    default_prob : float
        Predicted default probability (0-100)
    low_threshold : float
        Upper bound for LOW RISK (default: 40)
    high_threshold : float
        Upper bound for MODERATE RISK (default: 70)

    Returns
    -------
    tuple[str, str]
        (risk_label, risk_color)
    """
    try:
        probability = float(default_prob)
        low_bound = float(low_threshold)
        high_bound = float(high_threshold)
    except (TypeError, ValueError):
        return "NOT APPLICABLE", "#6b7280"

    if not isfinite(probability) or not isfinite(low_bound) or not isfinite(high_bound):
        return "NOT APPLICABLE", "#6b7280"

    if low_bound > high_bound:
        low_bound, high_bound = high_bound, low_bound

    if probability < low_bound:
        label = "LOW RISK"
    elif probability <= high_bound:
        label = "MODERATE RISK"
    else:
        label = "HIGH RISK"
    return label, RISK_TIER_COLORS[label]


def is_override(decision: str, risk_tier: str) -> bool:
    """True if a human decision contradicts the model's risk assessment."""
    return (
        (decision == "Approved" and risk_tier == "HIGH RISK")
        or (decision == "Rejected" and risk_tier == "LOW RISK")
    )