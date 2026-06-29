"""
model_utils.py
--------------
Handles all machine-learning concerns:
  - Loading the trained pipeline and transformer from disk
  - Feature alignment and preparation for inference
  - Prediction (probability output)
  - SHAP-based local explanations
  - Global feature importance
  - Model hyperparameter introspection
  - Static evaluation metrics (accuracy, confusion matrix, etc.)
  - Historical EDA / association-rule data for the diagnostics page

"""

import os

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_credit_models():
    """
    Loads the champion Random Forest pipeline and the Yeo-Johnson transformer
    from the sibling ``model/`` directory.

    Returns
    -------
    rf_model : sklearn Pipeline
    yj_transformer : sklearn PowerTransformer
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(project_root, "model")

    rf_path = os.path.join(model_dir, "champion_rf_pipeline.joblib")
    yj_path = os.path.join(model_dir, "yeo_johnson_transformer.joblib")

    if not os.path.exists(rf_path):
        st.error(
            f"🛑 CRITICAL ERROR: Cannot find the Random Forest model!\n\n"
            f"Python is searching exactly here: `{rf_path}`"
        )
        st.stop()

    if not os.path.exists(yj_path):
        st.error(
            f"🛑 CRITICAL ERROR: Cannot find the Transformer model!\n\n"
            f"Python is searching exactly here: `{yj_path}`"
        )
        st.stop()

    rf_model = joblib.load(rf_path)
    yj_transformer = joblib.load(yj_path)

    return rf_model, yj_transformer


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(app_row, rf_model):
    """
    Converts a raw applicant row (pandas Series) into a feature DataFrame
    that is aligned with the columns the pipeline was trained on.

    Parameters
    ----------
    app_row  : pd.Series  – one applicant record
    rf_model : sklearn Pipeline

    Returns
    -------
    features_aligned : pd.DataFrame  (1 row)
    """
    drop_cols = ["ID", "UNNAMED: 0", "DEFAULT PAYMENT NEXT MONTH", "Y"]
    features_raw = app_row.drop(labels=drop_cols, errors="ignore")

    features_numeric = pd.to_numeric(features_raw, errors="coerce")
    features_df = pd.DataFrame([features_numeric])

    # Override history columns for brand-new applicants
    if str(app_row.get("IS_NEW_APPLICANT", "False")).strip().lower() in [
        "true", "1", "1.0", "yes", "t"
    ]:
        for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
            features_df[col] = -2
        for i in range(1, 7):
            features_df[f"BILL_AMT{i}"] = 0
            features_df[f"PAY_AMT{i}"] = 0

    # Conservative fill-ins for common demographic fields
    features_df["AGE"] = features_df.get("AGE", pd.Series([35])).fillna(35)
    features_df["LIMIT_BAL"] = features_df.get("LIMIT_BAL", pd.Series([100_000])).fillna(100_000)
    features_df["EDUCATION"] = features_df.get("EDUCATION", pd.Series([4])).fillna(4)
    features_df["MARRIAGE"] = features_df.get("MARRIAGE", pd.Series([3])).fillna(3)

    features_df = features_df.fillna(0)

    if hasattr(rf_model, "feature_names_in_"):
        features_aligned = features_df.reindex(
            columns=rf_model.feature_names_in_, fill_value=0
        )
    else:
        features_aligned = features_df

    return features_aligned


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def process_and_predict(app_row, rf_model, yj_transformer=None):
    """
    Returns the default probability (0–100) for a single applicant row.

    The pipeline handles all internal transformations; ``yj_transformer``
    is accepted for API compatibility but is not applied here directly.
    """
    features_final = prepare_features(app_row, rf_model)
    probs = rf_model.predict_proba(features_final)
    return probs[0][1] * 100


# ---------------------------------------------------------------------------
# SHAP – local explanations
# ---------------------------------------------------------------------------

def get_risk_factors(app_row, rf_model, yj_transformer=None):
    """
    Uses SHAP TreeExplainer to rank the top 5 feature contributions for a
    single applicant.  Returns a DataFrame sorted ascending by contribution
    (negative = protective, positive = risk-increasing).

    Parameters
    ----------
    app_row        : pd.Series
    rf_model       : sklearn Pipeline
    yj_transformer : ignored (kept for API compatibility)

    Returns
    -------
    pd.DataFrame with columns: Feature, Contribution, Abs_Contribution
    """
    features_final = prepare_features(app_row, rf_model)

    # Push data through all preprocessing steps, extract the estimator
    Xt = features_final
    if hasattr(rf_model, "steps"):
        for _, transformer in rf_model.steps[:-1]:
            Xt = transformer.transform(Xt)
        estimator = rf_model.steps[-1][1]
    else:
        estimator = rf_model

    explainer = shap.TreeExplainer(estimator)
    shap_values_raw = explainer.shap_values(Xt)

    # Version-proof SHAP extraction
    vals = shap_values_raw.values if hasattr(shap_values_raw, "values") else shap_values_raw
    if isinstance(vals, list):
        vals = vals[1]                     # binary classification: positive class
    vals = np.array(vals)
    if len(vals.shape) == 3:
        vals = vals[:, :, 1]              # (n_samples, n_features, n_classes)

    contributions = [float(x) for x in np.array(vals[0]).flatten()]

    # Resolve feature names after preprocessing
    if hasattr(estimator, "feature_names_in_"):
        expected_cols = estimator.feature_names_in_
    elif hasattr(Xt, "columns"):
        expected_cols = Xt.columns.tolist()
    else:
        expected_cols = features_final.columns.tolist()

    df = pd.DataFrame(
        [{"Feature": expected_cols[i], "Contribution": contributions[i]}
         for i in range(len(expected_cols))]
    )

    df["Abs_Contribution"] = df["Contribution"].abs()
    df = df.sort_values("Abs_Contribution", ascending=False).head(5)

    # Strip pipeline prefixes  (e.g. 'remainder__BILL_AMT1' → 'BILL_AMT1')
    df["Feature"] = df["Feature"].str.split("__").str[-1]
    df["Feature"] = df["Feature"].map(lambda x: _FEATURE_LABELS.get(x, x))

    return df.sort_values("Contribution", ascending=True)


# ---------------------------------------------------------------------------
# Global feature importance
# ---------------------------------------------------------------------------

def get_global_feature_importance(rf_model):
    """
    Returns a DataFrame of the top-10 features by built-in RF importance,
    sorted ascending (for horizontal bar charts).
    """
    estimator = rf_model.steps[-1][1] if hasattr(rf_model, "steps") else rf_model
    importances = estimator.feature_importances_

    if hasattr(estimator, "feature_names_in_"):
        expected_cols = estimator.feature_names_in_
    else:
        expected_cols = [f"Feature {i}" for i in range(len(importances))]

    df = pd.DataFrame({"Feature": expected_cols, "Importance": importances})
    df["Feature"] = df["Feature"].str.split("__").str[-1]
    df["Feature"] = df["Feature"].map(lambda x: _FEATURE_LABELS.get(x, x))

    return df.sort_values("Importance", ascending=True).tail(10)


# ---------------------------------------------------------------------------
# Model introspection helpers
# ---------------------------------------------------------------------------

def get_model_params(rf_model):
    """Returns a dict of core hyperparameters for the RF estimator."""
    estimator = rf_model.steps[-1][1] if hasattr(rf_model, "steps") else rf_model
    params = estimator.get_params()

    return {
        "Number of Estimators (Trees)": params.get("n_estimators", "N/A"),
        "Max Depth": params.get("max_depth", "None (Unlimited)"),
        "Min Samples Split": params.get("min_samples_split", "N/A"),
        "Criterion": str(params.get("criterion", "N/A")).capitalize(),
    }


def get_model_metrics():
    """
    Returns static KPIs and a confusion matrix from the model's test phase.
    Update these values to match your final Jupyter Notebook output.

    Returns
    -------
    kpis : dict
    cm   : list[list[int]]  – [[TN, FP], [FN, TP]]
    """
    kpis = {
        "Accuracy": "82.4%",
        "Precision": "76.1%",
        "Recall": "68.3%",
        "F1-Score": "72.0%",
    }

    cm = [
        [4200, 350],   # Actual Paid (0)    → [TN, FP]
        [600, 850],    # Actual Default (1) → [FN, TP]
    ]

    return kpis, cm


# ---------------------------------------------------------------------------
# EDA / association-rule data
# ---------------------------------------------------------------------------

def get_historical_eda_data():
    """
    Supplies aggregated macro-level statistics from the training dataset for
    the EDA charts on the diagnostics page.

    Returns
    -------
    education_data : pd.DataFrame  – default rate by education level
    limit_data     : dict          – credit-limit distribution by outcome
    """
    education_data = pd.DataFrame({
        "Education": ["Graduate School", "University", "High School", "Others"],
        "Default_Rate": [19.2, 23.7, 25.1, 7.0],
    })

    limit_data = {
        "Paid":      [50_000, 100_000, 150_000, 250_000, 500_000],
        "Defaulted": [20_000,  50_000,  90_000, 150_000, 300_000],
    }

    return education_data, limit_data


def get_association_rules():
    """
    Returns the top association rules mined via Apriori/FP-Growth.
    Update these to match your actual Jupyter Notebook output.
    """
    rules = [
        {
            "Antecedent (Condition)": "PAY_0 = Delay 2+ Months",
            "Consequent (Outcome)": "Risk = Default",
            "Support": "11.2%", "Confidence": "69.4%", "Lift": "3.14",
        },
        {
            "Antecedent (Condition)": "LIMIT_BAL < $50k AND EDUCATION = High School",
            "Consequent (Outcome)": "Risk = Default",
            "Support": "8.5%", "Confidence": "45.2%", "Lift": "2.05",
        },
        {
            "Antecedent (Condition)": "PAY_0 = Paid Duly AND PAY_2 = Paid Duly",
            "Consequent (Outcome)": "Risk = Low",
            "Support": "28.3%", "Confidence": "88.1%", "Lift": "1.12",
        },
        {
            "Antecedent (Condition)": "AGE < 25 AND LIMIT_BAL < $30k",
            "Consequent (Outcome)": "Risk = Default",
            "Support": "6.1%", "Confidence": "39.8%", "Lift": "1.80",
        },
    ]
    return pd.DataFrame(rules)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def get_payment_trend(app_row):
    """
    Transforms a raw applicant row into a 6-month time-series DataFrame
    suitable for the payment-history line chart.
    """
    months = ["5 Mo Ago", "4 Mo Ago", "3 Mo Ago", "2 Mo Ago", "1 Mo Ago", "Current"]
    bills = [pd.to_numeric(app_row.get(f"BILL_AMT{i}", 0), errors="coerce") for i in range(6, 0, -1)]
    pays  = [pd.to_numeric(app_row.get(f"PAY_AMT{i}",  0), errors="coerce") for i in range(6, 0, -1)]
    return pd.DataFrame({"Month": months, "Billed Amount": bills, "Paid Amount": pays})


# Human-readable labels shared by local and global importance functions
_FEATURE_LABELS = {
    "LIMIT_BAL": "Total Credit Limit",
    "SEX": "Gender",
    "EDUCATION": "Education Level",
    "MARRIAGE": "Marital Status",
    "AGE": "Applicant Age",
    "PAY_0": "Current Payment Status",
    "PAY_2": "Payment Status (2 Mo Ago)",
    "PAY_3": "Payment Status (3 Mo Ago)",
    "PAY_4": "Payment Status (4 Mo Ago)",
    "PAY_5": "Payment Status (5 Mo Ago)",
    "PAY_6": "Payment Status (6 Mo Ago)",
    "BILL_AMT1": "Current Billed Amount",
    "BILL_AMT2": "Billed Amount (2 Mo Ago)",
    "BILL_AMT3": "Billed Amount (3 Mo Ago)",
    "BILL_AMT4": "Billed Amount (4 Mo Ago)",
    "BILL_AMT5": "Billed Amount (5 Mo Ago)",
    "BILL_AMT6": "Billed Amount (6 Mo Ago)",
    "PAY_AMT1": "Current Paid Amount",
    "PAY_AMT2": "Paid Amount (2 Mo Ago)",
    "PAY_AMT3": "Paid Amount (3 Mo Ago)",
    "PAY_AMT4": "Paid Amount (4 Mo Ago)",
    "PAY_AMT5": "Paid Amount (5 Mo Ago)",
    "PAY_AMT6": "Paid Amount (6 Mo Ago)",
}