"""
All data-wrangling helpers that are NOT model-specific:
  - Column name mapping (X1-X23 → feature names)
  - Dataset schema repair (inject safe defaults for missing columns)
  - CSV/Excel header cleaning (staggered or duplicate headers)
  - Primary-key / ID column auto-detection
  - Categorical value display mappers (SEX, EDUCATION, MARRIAGE)
  - Payment-status badge renderer
  - New-applicant auto-detection
"""

import numpy as np
import pandas as pd

from constants import PAY_COLS, BILL_COLS, PAY_AMT_COLS


# ---------------------------------------------------------------------------
# Column-name mapping
# ---------------------------------------------------------------------------

#: Maps raw X1-X23 column names (as exported by the Taiwan dataset) to the
#: human-readable feature names expected by the model pipeline.
_COLUMN_MAP = {
    "X1": "LIMIT_BAL", "X2": "SEX",      "X3": "EDUCATION", "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",     "X7": "PAY_2",    "X8": "PAY_3",     "X9": "PAY_4",
    "X10": "PAY_5",    "X11": "PAY_6",
    "X12": "BILL_AMT1","X13": "BILL_AMT2","X14": "BILL_AMT3",
    "X15": "BILL_AMT4","X16": "BILL_AMT5","X17": "BILL_AMT6",
    "X18": "PAY_AMT1", "X19": "PAY_AMT2", "X20": "PAY_AMT3",
    "X21": "PAY_AMT4", "X22": "PAY_AMT5", "X23": "PAY_AMT6",
}


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames raw X1–X23 columns to their actual feature names.
    Columns that are already correctly named are left untouched.
    """
    return df.rename(columns=_COLUMN_MAP)


# ---------------------------------------------------------------------------
# Schema repair
# ---------------------------------------------------------------------------

#: Conservative statistical baselines injected when a required column is absent.
_SAFE_DEFAULTS = {
    "LIMIT_BAL": 50_000,
    "SEX": 2,          # 2 = Female (or mode of dataset)
    "EDUCATION": 4,    # 4 = Others/Unknown
    "MARRIAGE": 3,     # 3 = Others/Unknown
    "AGE": 35,         # Median age
    **{col: 0 for col in PAY_COLS},
    **{col: 0 for col in BILL_COLS},
    **{col: 0 for col in PAY_AMT_COLS},
}


def repair_dataset_schema(df: pd.DataFrame):
    """
    Injects conservative baseline values for any required feature column that
    is missing from the uploaded dataset, rather than rejecting the file.

    Parameters
    ----------
    df : pd.DataFrame  – the uploaded dataset after column mapping

    Returns
    -------
    df              : pd.DataFrame  – the dataset with missing columns filled
    missing_detected : list[str]    – names of columns that were injected
    """
    missing_detected = []

    for col, default_val in _SAFE_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default_val
            missing_detected.append(col)

    return df, missing_detected


# ---------------------------------------------------------------------------
# Header cleaning
# ---------------------------------------------------------------------------

def clean_dataset_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects and corrects two common malformed-CSV patterns:

    1. **Duplicate header row** – the first data row is identical to the
       column names (e.g. exported from Excel with a repeated header).
    2. **Staggered header** – columns are named X1, X2 … but row 0 contains
       the real names (LIMIT_BAL, ID, AGE …).

    Returns the repaired DataFrame with numeric columns cast to numeric types.
    """
    if df.empty:
        return df

    def _coerce_numeric(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Converts columns to numeric where possible; leaves text columns intact."""
        for col in dataframe.columns:
            try:
                dataframe[col] = pd.to_numeric(dataframe[col])
            except (ValueError, TypeError):
                pass
        return dataframe

    # Pattern 1: duplicate header row
    if all(str(c).strip() == str(v).strip() for c, v in zip(df.columns, df.iloc[0])):
        return _coerce_numeric(df.iloc[1:].reset_index(drop=True))

    # Pattern 2: staggered header – row 0 contains real column names
    first_row = df.iloc[0].astype(str).str.strip().str.upper().values
    if any(token in first_row for token in ("LIMIT_BAL", "ID", "AGE")):
        df.columns = df.iloc[0].astype(str).str.strip()
        df = df.iloc[1:].reset_index(drop=True)
        return _coerce_numeric(df)

    return df


# ---------------------------------------------------------------------------
# ID-column detection
# ---------------------------------------------------------------------------

def detect_id_column(df: pd.DataFrame):
    """
    Heuristically identifies the primary-key / ID column in a DataFrame.

    Resolution order
    ----------------
    1. Exact match against a list of well-known ID column names.
    2. Any column whose name contains "id" or "num" and whose values are
       100 % unique.
    3. The very first column, if all its values are unique.

    Returns
    -------
    str | None  – column name, or ``None`` if no candidate was found.
    """
    known_id_names = {"id", "clientnum", "client_id", "customer_id",
                      "applicant_id", "ref_no"}

    for col in df.columns:
        if str(col).lower() in known_id_names:
            return col

    for col in df.columns:
        col_lower = str(col).lower()
        if ("id" in col_lower or "num" in col_lower) and df[col].nunique() == len(df):
            return col

    if df.iloc[:, 0].nunique() == len(df):
        return df.columns[0]

    return None


# ---------------------------------------------------------------------------
# New-applicant detection
# ---------------------------------------------------------------------------

def detect_new_applicants(df: pd.DataFrame) -> pd.Series:
    """
    Infers whether each row represents a brand-new applicant (i.e. one with
    no prior payment or billing history) and returns a boolean Series.

    An applicant is considered new if ALL their payment-status columns contain
    only -2 (No Consumption), 0, or NaN **and** all their financial-history
    columns are 0 or NaN.
    """
    temp_pay = df[PAY_COLS].apply(pd.to_numeric, errors="coerce")
    temp_fin = df[BILL_COLS + PAY_AMT_COLS].apply(pd.to_numeric, errors="coerce")

    is_status_empty     = temp_pay.isin([-2, 0, np.nan]).all(axis=1)
    is_financials_empty = temp_fin.isin([0, np.nan]).all(axis=1)

    return is_status_empty & is_financials_empty


# ---------------------------------------------------------------------------
# Categorical display mappers
# ---------------------------------------------------------------------------

def map_sex(val) -> str:
    """Maps the numeric SEX code to a human-readable label."""
    return {1: "Male", 2: "Female"}.get(val, "Unknown")


def map_education(val) -> str:
    """Maps the numeric EDUCATION code to a human-readable label."""
    return {1: "Graduate School", 2: "University", 3: "High School"}.get(val, "Others")


def map_marriage(val) -> str:
    """Maps the numeric MARRIAGE code to a human-readable label."""
    return {1: "Married", 2: "Single", 3: "Others"}.get(val, "Unknown")


# ---------------------------------------------------------------------------
# Payment-status badge renderer
# ---------------------------------------------------------------------------

def get_history_badge(val) -> tuple[str, str]:
    """
    Returns a ``(css_class, label)`` tuple for rendering a payment-status
    badge in the applicant profile panel.

    CSS classes correspond to the badge styles defined in ``style.css``:
      - ``badge-gray``  – neutral / no consumption
      - ``badge-navy``  – paid on time
      - ``badge-red``   – payment delayed
    """
    try:
        val = int(val)
        if val == -2:
            return "badge-gray", "No Consumption (-2)"
        if val == -1:
            return "badge-navy", "Paid in Full (-1)"
        if val == 0:
            return "badge-navy", "Revolving (0)"
        return "badge-red", f"{val} Month(s) Late"
    except (ValueError, TypeError):
        return "badge-gray", "Unknown"