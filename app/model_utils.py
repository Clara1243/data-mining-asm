import os
import pandas as pd
import streamlit as st
import joblib
import shap
import numpy as np

def load_credit_models():
    # 1. Get the directory where model_utils.py lives (the 'app' folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Step UP one level to the project root, then DOWN into the 'model' folder
    project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(project_root, "model")
    
    # 3. Define the exact paths
    rf_path = os.path.join(model_dir, "champion_rf_pipeline.joblib")
    yj_path = os.path.join(model_dir, "yeo_johnson_transformer.joblib")
    
    # 4. Safety Check
    if not os.path.exists(rf_path):
        st.error(f"🛑 CRITICAL ERROR: Cannot find the Random Forest model!\n\nPython is searching exactly here: `{rf_path}`")
        st.stop()
        
    if not os.path.exists(yj_path):
        st.error(f"🛑 CRITICAL ERROR: Cannot find the Transformer model!\n\nPython is searching exactly here: `{yj_path}`")
        st.stop()

    # 5. Load models if paths are valid
    rf_model = joblib.load(rf_path)
    yj_transformer = joblib.load(yj_path)
    
    return rf_model, yj_transformer

def map_columns(df):
    """Maps raw X1-X23 columns from the uploaded dataset to actual feature names."""
    mapping = {
        'X1': 'LIMIT_BAL', 'X2': 'SEX', 'X3': 'EDUCATION', 'X4': 'MARRIAGE', 
        'X5': 'AGE', 'X6': 'PAY_0', 'X7': 'PAY_2', 'X8': 'PAY_3', 'X9': 'PAY_4', 
        'X10': 'PAY_5', 'X11': 'PAY_6', 'X12': 'BILL_AMT1', 'X13': 'BILL_AMT2', 
        'X14': 'BILL_AMT3', 'X15': 'BILL_AMT4', 'X16': 'BILL_AMT5', 'X17': 'BILL_AMT6', 
        'X18': 'PAY_AMT1', 'X19': 'PAY_AMT2', 'X20': 'PAY_AMT3', 
        'X21': 'PAY_AMT4', 'X22': 'PAY_AMT5', 'X23': 'PAY_AMT6'
    }
    return df.rename(columns=mapping)

def validate_dataset_schema(df):
    """Checks if the uploaded dataset contains the critical minimum features."""
    required_features = [
        'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'
    ]
    
    df_columns = df.columns.tolist()
    missing_features = [feature for feature in required_features if feature not in df_columns]
    
    if missing_features:
        return False, missing_features
    return True, []

def prepare_features(app_row, rf_model):
    """Cleans data and automatically aligns it to the pipeline's exact expected features."""
    drop_cols = ['ID', 'UNNAMED: 0', 'DEFAULT PAYMENT NEXT MONTH', 'Y']
    features_raw = app_row.drop(labels=drop_cols, errors='ignore')
    
    # Convert all to numeric
    features_numeric = pd.to_numeric(features_raw, errors='coerce').fillna(0)
    features_df = pd.DataFrame([features_numeric])
    
    # Ultimate source of truth: Align perfectly with what the pipeline expects
    if hasattr(rf_model, 'feature_names_in_'):
        expected_cols = rf_model.feature_names_in_
        # Reindex automatically drops extra columns and fills missing ones with 0
        features_aligned = features_df.reindex(columns=expected_cols, fill_value=0)
    else:
        features_aligned = features_df
        
    return features_aligned

def process_and_predict(app_row, rf_model, yj_transformer=None):
    """Calculates the probability using the end-to-end Random Forest pipeline."""
    # We pass the raw aligned features. The pipeline handles its own internal transformations!
    features_final = prepare_features(app_row, rf_model)
    
    probs = rf_model.predict_proba(features_final)
    return probs[0][1] * 100

def get_payment_trend(app_row):
    """Transforms raw row into a time-series DataFrame for the line chart."""
    months = ['5 Mo Ago', '4 Mo Ago', '3 Mo Ago', '2 Mo Ago', '1 Mo Ago', 'Current']
    bills = [pd.to_numeric(app_row.get(f'BILL_AMT{i}', 0), errors='coerce') for i in range(6, 0, -1)]
    pays = [pd.to_numeric(app_row.get(f'PAY_AMT{i}', 0), errors='coerce') for i in range(6, 0, -1)]
    return pd.DataFrame({'Month': months, 'Billed Amount': bills, 'Paid Amount': pays})

def get_risk_factors(app_row, rf_model, yj_transformer=None):
    """Uses SHAP to explain the Random Forest decision for the UI chart."""
    features_final = prepare_features(app_row, rf_model)
    
    # Push the data through all preprocessing steps first
    Xt = features_final
    if hasattr(rf_model, 'steps'):
        for name, transformer in rf_model.steps[:-1]:
            Xt = transformer.transform(Xt)
        estimator = rf_model.steps[-1][1]
    else:
        estimator = rf_model
        
    explainer = shap.TreeExplainer(estimator)
    shap_values_raw = explainer.shap_values(Xt)
    
    # --- THE ULTIMATE FIX: SHAP Version-Proof Extraction ---
    # 1. Handle newer SHAP Explanation objects
    if hasattr(shap_values_raw, 'values'):
        vals = shap_values_raw.values
    else:
        vals = shap_values_raw
        
    # 2. Handle older List format (binary classification)
    if isinstance(vals, list):
        vals = vals[1]  # Extract positive class
        
    # 3. Convert whatever is left into a strict numpy array
    vals = np.array(vals)
    
    # 4. If SHAP returned a 3D array: (n_samples, n_features, n_classes)
    if len(vals.shape) == 3:
        vals = vals[:, :, 1]  # Extract positive class
        
    # 5. Extract the single applicant's row and force it into standard floats
    contributions = [float(x) for x in np.array(vals[0]).flatten()]
    # -----------------------------------------------------------
    
    if hasattr(estimator, 'feature_names_in_'):
        expected_cols = estimator.feature_names_in_
    elif hasattr(Xt, 'columns'):
        expected_cols = Xt.columns.tolist()
    else:
        expected_cols = features_final.columns.tolist()
        
    factors = [{'Feature': expected_cols[i], 'Contribution': contributions[i]} for i in range(len(expected_cols))]
    df = pd.DataFrame(factors)
    
    # Pandas can now safely sort this because it's guaranteed to be standard floats
    df['Abs_Contribution'] = df['Contribution'].abs()
    df = df.sort_values(by='Abs_Contribution', ascending=False).head(5)
    
    # Clean up complex names if the pipeline altered them (e.g., 'remainder__BILL_AMT1')
    df['Feature'] = df['Feature'].str.split('__').str[-1]
    
    # Human-readable dictionary mapping
    feature_dictionary = {
        'LIMIT_BAL': 'Total Credit Limit', 'SEX': 'Gender', 'EDUCATION': 'Education Level',
        'MARRIAGE': 'Marital Status', 'AGE': 'Applicant Age', 'PAY_0': 'Current Payment Status',
        'PAY_2': 'Payment Status (2 Mo Ago)', 'PAY_3': 'Payment Status (3 Mo Ago)',
        'PAY_4': 'Payment Status (4 Mo Ago)', 'PAY_5': 'Payment Status (5 Mo Ago)',
        'PAY_6': 'Payment Status (6 Mo Ago)', 'BILL_AMT1': 'Current Billed Amount',
        'BILL_AMT2': 'Billed Amount (2 Mo Ago)', 'BILL_AMT3': 'Billed Amount (3 Mo Ago)',
        'BILL_AMT4': 'Billed Amount (4 Mo Ago)', 'BILL_AMT5': 'Billed Amount (5 Mo Ago)',
        'BILL_AMT6': 'Billed Amount (6 Mo Ago)', 'PAY_AMT1': 'Current Paid Amount',
        'PAY_AMT2': 'Paid Amount (2 Mo Ago)', 'PAY_AMT3': 'Paid Amount (3 Mo Ago)',
        'PAY_AMT4': 'Paid Amount (4 Mo Ago)', 'PAY_AMT5': 'Paid Amount (5 Mo Ago)',
        'PAY_AMT6': 'Paid Amount (6 Mo Ago)'
    }
    
    df['Feature'] = df['Feature'].map(lambda x: feature_dictionary.get(x, x))
    return df.sort_values(by='Contribution', ascending=True)

def get_model_params(rf_model):
    """Extracts core hyperparameters from the Random Forest estimator."""
    estimator = rf_model.steps[-1][1] if hasattr(rf_model, 'steps') else rf_model
    params = estimator.get_params()
    
    return {
        "Number of Estimators (Trees)": params.get('n_estimators', 'N/A'),
        "Max Depth": params.get('max_depth', 'None (Unlimited)'),
        "Min Samples Split": params.get('min_samples_split', 'N/A'),
        "Criterion": params.get('criterion', 'N/A').capitalize()
    }

def get_global_feature_importance(rf_model):
    """Extracts the built-in global feature importance from the Random Forest."""
    estimator = rf_model.steps[-1][1] if hasattr(rf_model, 'steps') else rf_model
    importances = estimator.feature_importances_
    
    if hasattr(estimator, 'feature_names_in_'):
        expected_cols = estimator.feature_names_in_
    else:
        expected_cols = [f"Feature {i}" for i in range(len(importances))]
        
    df = pd.DataFrame({'Feature': expected_cols, 'Importance': importances})
    
    df['Feature'] = df['Feature'].str.split('__').str[-1]
    
    feature_dictionary = {
        'LIMIT_BAL': 'Total Credit Limit', 'SEX': 'Gender', 'EDUCATION': 'Education Level',
        'MARRIAGE': 'Marital Status', 'AGE': 'Applicant Age', 'PAY_0': 'Current Payment Status',
        'PAY_2': 'Payment Status (2 Mo Ago)', 'PAY_3': 'Payment Status (3 Mo Ago)',
        'PAY_4': 'Payment Status (4 Mo Ago)', 'PAY_5': 'Payment Status (5 Mo Ago)',
        'PAY_6': 'Payment Status (6 Mo Ago)', 'BILL_AMT1': 'Current Billed Amount',
        'BILL_AMT2': 'Billed Amount (2 Mo Ago)', 'BILL_AMT3': 'Billed Amount (3 Mo Ago)',
        'BILL_AMT4': 'Billed Amount (4 Mo Ago)', 'BILL_AMT5': 'Billed Amount (5 Mo Ago)',
        'BILL_AMT6': 'Billed Amount (6 Mo Ago)', 'PAY_AMT1': 'Current Paid Amount',
        'PAY_AMT2': 'Paid Amount (2 Mo Ago)', 'PAY_AMT3': 'Paid Amount (3 Mo Ago)',
        'PAY_AMT4': 'Paid Amount (4 Mo Ago)', 'PAY_AMT5': 'Paid Amount (5 Mo Ago)',
        'PAY_AMT6': 'Paid Amount (6 Mo Ago)'
    }
    
    df['Feature'] = df['Feature'].map(lambda x: feature_dictionary.get(x, x))
    return df.sort_values(by='Importance', ascending=True).tail(10)

def get_model_metrics():
    """
    Returns static evaluation metrics from the model's testing phase.
    UPDATE THESE VALUES to match the exact output of your final Jupyter Notebook.
    """
    kpis = {
        "Accuracy": "82.4%",
        "Precision": "76.1%",
        "Recall": "68.3%",
        "F1-Score": "72.0%"
    }
    
    cm = [
        [4200, 350],  # Actual Paid (0)
        [600, 850]    # Actual Default (1)
    ]
    
    return kpis, cm

def get_historical_eda_data():
    """
    Supplies aggregated macro-level data from the training dataset for EDA charts.
    """
    # 1. Default Rate by Education Level
    education_data = pd.DataFrame({
        'Education': ['Graduate School', 'University', 'High School', 'Others'],
        'Default_Rate': [19.2, 23.7, 25.1, 7.0] # Standard Taiwan dataset metrics
    })
    
    # 2. Credit Limit Distribution by Status (Approximated quartiles)
    limit_data = {
        'Paid': [50000, 100000, 150000, 250000, 500000],
        'Defaulted': [20000, 50000, 90000, 150000, 300000]
    }
    
    return education_data, limit_data

def get_association_rules():
    """
    Returns the top Association Rules mined via Apriori/FP-Growth.
    UPDATE THESE to match your actual Jupyter Notebook output.
    """
    rules = [
        {
            "Antecedent (Condition)": "PAY_0 = Delay 2+ Months", 
            "Consequent (Outcome)": "Risk = Default", 
            "Support": "11.2%", "Confidence": "69.4%", "Lift": "3.14"
        },
        {
            "Antecedent (Condition)": "LIMIT_BAL < $50k AND EDUCATION = High School", 
            "Consequent (Outcome)": "Risk = Default", 
            "Support": "8.5%", "Confidence": "45.2%", "Lift": "2.05"
        },
        {
            "Antecedent (Condition)": "PAY_0 = Paid Duly AND PAY_2 = Paid Duly", 
            "Consequent (Outcome)": "Risk = Low", 
            "Support": "28.3%", "Confidence": "88.1%", "Lift": "1.12"
        },
        {
            "Antecedent (Condition)": "AGE < 25 AND LIMIT_BAL < $30k", 
            "Consequent (Outcome)": "Risk = Default", 
            "Support": "6.1%", "Confidence": "39.8%", "Lift": "1.80"
        }
    ]
    return pd.DataFrame(rules)