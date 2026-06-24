from pathlib import Path
import pandas as pd
import xgboost as xgb

def load_credit_model(model_path=Path("model") / "xgboost_credit_model.json"):
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    return model

def map_columns(df):
    """Maps X1-X23 to the actual feature names the model expects."""
    mapping = {
        'X1': 'LIMIT_BAL', 'X2': 'SEX', 'X3': 'EDUCATION', 'X4': 'MARRIAGE', 
        'X5': 'AGE', 'X6': 'PAY_0', 'X7': 'PAY_2', 'X8': 'PAY_3', 'X9': 'PAY_4', 
        'X10': 'PAY_5', 'X11': 'PAY_6', 'X12': 'BILL_AMT1', 'X13': 'BILL_AMT2', 
        'X14': 'BILL_AMT3', 'X15': 'BILL_AMT4', 'X16': 'BILL_AMT5', 'X17': 'BILL_AMT6', 
        'X18': 'PAY_AMT1', 'X19': 'PAY_AMT2', 'X20': 'PAY_AMT3', 
        'X21': 'PAY_AMT4', 'X22': 'PAY_AMT5', 'X23': 'PAY_AMT6'
    }
    # Update mapping if your file has X18-X23 for PAY_AMT
    return df.rename(columns=mapping)

def process_and_predict(app_row, model):
    # 1. Drop artifacts
    drop_cols = ['ID', 'UNNAMED: 0', 'DEFAULT PAYMENT NEXT MONTH', 'Y']
    features_raw = app_row.drop(labels=drop_cols, errors='ignore')
    
    # 2. Force Numeric
    features_numeric = pd.to_numeric(features_raw, errors='coerce').fillna(0)
    features_df = pd.DataFrame([features_numeric])
    
    # 3. Align with Model Metadata
    expected_cols = model.get_booster().feature_names
    features_final = features_df.reindex(columns=expected_cols, fill_value=0).astype('float32')
    
    # 4. Predict
    dtest = xgb.DMatrix(features_final, feature_names=expected_cols)
    probs = model.get_booster().predict(dtest)
    
    return probs[0] * 100