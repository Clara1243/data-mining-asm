import os
import pandas as pd
import xgboost as xgb

# Notice we changed the default path here to include 'model/'
def load_credit_model(model_filename="model/xgboost_credit_model.json"):
    # 1. Get the exact folder path where model_utils.py lives (the 'app' folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Join it with the filename to get the absolute path
    # This will result in something like: C:\...\app\model\xgboost_credit_model.json
    model_path = os.path.join(current_dir, model_filename)
    
    # 3. Load the model using this absolute path
    model = xgb.XGBClassifier()
    model.load_model(model_path)
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
    return df.rename(columns=mapping)

def prepare_features(app_row, model):
    """Isolates the data cleaning and alignment logic."""
    drop_cols = ['ID', 'UNNAMED: 0', 'DEFAULT PAYMENT NEXT MONTH', 'Y']
    features_raw = app_row.drop(labels=drop_cols, errors='ignore')
    features_numeric = pd.to_numeric(features_raw, errors='coerce').fillna(0)
    features_df = pd.DataFrame([features_numeric])
    expected_cols = model.get_booster().feature_names
    return features_df.reindex(columns=expected_cols, fill_value=0).astype('float32')

def process_and_predict(app_row, model):
    features_final = prepare_features(app_row, model)
    expected_cols = features_final.columns.tolist()
    dtest = xgb.DMatrix(features_final, feature_names=expected_cols)
    probs = model.get_booster().predict(dtest)
    return probs[0] * 100

def get_payment_trend(app_row):
    """Transforms raw row into a time-series DataFrame for the line chart."""
    months = ['5 Mo Ago', '4 Mo Ago', '3 Mo Ago', '2 Mo Ago', '1 Mo Ago', 'Current']
    
    # Extract chronologically (6 is oldest, 1 is newest)
    bills = [pd.to_numeric(app_row.get(f'BILL_AMT{i}', 0), errors='coerce') for i in range(6, 0, -1)]
    pays = [pd.to_numeric(app_row.get(f'PAY_AMT{i}', 0), errors='coerce') for i in range(6, 0, -1)]
    
    return pd.DataFrame({'Month': months, 'Billed Amount': bills, 'Paid Amount': pays})

def get_risk_factors(app_row, model):
    """Calculates feature contributions and translates them to readable names."""
    features_final = prepare_features(app_row, model)
    expected_cols = features_final.columns.tolist()
    dtest = xgb.DMatrix(features_final, feature_names=expected_cols)
    
    contribs = model.get_booster().predict(dtest, pred_contribs=True)[0]
    
    factors = [{'Feature': expected_cols[i], 'Contribution': contribs[i]} for i in range(len(expected_cols))]
    df = pd.DataFrame(factors)
    
    df['Abs_Contribution'] = df['Contribution'].abs()
    df = df.sort_values(by='Abs_Contribution', ascending=False).head(5)
    
    # --- NEW: Translate technical names to human-readable descriptions ---
    feature_dictionary = {
        'LIMIT_BAL': 'Total Credit Limit',
        'SEX': 'Gender',
        'EDUCATION': 'Education Level',
        'MARRIAGE': 'Marital Status',
        'AGE': 'Applicant Age',
        'PAY_0': 'Current Payment Status',
        'PAY_2': 'Payment Status (2 Mo Ago)',
        'PAY_3': 'Payment Status (3 Mo Ago)',
        'PAY_4': 'Payment Status (4 Mo Ago)',
        'PAY_5': 'Payment Status (5 Mo Ago)',
        'PAY_6': 'Payment Status (6 Mo Ago)',
        'BILL_AMT1': 'Current Billed Amount',
        'BILL_AMT2': 'Billed Amount (2 Mo Ago)',
        'BILL_AMT3': 'Billed Amount (3 Mo Ago)',
        'BILL_AMT4': 'Billed Amount (4 Mo Ago)',
        'BILL_AMT5': 'Billed Amount (5 Mo Ago)',
        'BILL_AMT6': 'Billed Amount (6 Mo Ago)',
        'PAY_AMT1': 'Current Paid Amount',
        'PAY_AMT2': 'Paid Amount (2 Mo Ago)',
        'PAY_AMT3': 'Paid Amount (3 Mo Ago)',
        'PAY_AMT4': 'Paid Amount (4 Mo Ago)',
        'PAY_AMT5': 'Paid Amount (5 Mo Ago)',
        'PAY_AMT6': 'Paid Amount (6 Mo Ago)'
    }
    
    # Map the readable names to the DataFrame, keeping original if not found
    df['Feature'] = df['Feature'].map(lambda x: feature_dictionary.get(x, x))
    
    return df.sort_values(by='Contribution', ascending=True)