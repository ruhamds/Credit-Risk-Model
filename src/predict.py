# src/predict.py
import pandas as pd
import joblib

# Load model and features with correct paths inside the container
model = joblib.load("model/best_customer_risk_model.pkl")
FEATURES_PATH = "model/features_train_woe.csv"
feature_columns = pd.read_csv(FEATURES_PATH).columns.tolist()

def predict_function(input_data):
    """
    input_data: Pydantic model instance (with .dict() method)
    Returns: tuple (prediction, risk_probability)
    """
    # Convert input_data (Pydantic) to DataFrame
    input_dict = input_data.dict()
    input_df = pd.DataFrame([{
        'Recency_woe': input_dict['recency'],
        'Frequency_woe': input_dict['frequency'],
        'Monetary_woe': input_dict['monetary']
    }])
    
    # Enforce column order as expected by the model
    input_df = input_df[feature_columns]
    
    # Predict
    prediction = model.predict(input_df)
    risk_proba = model.predict_proba(input_df)[:, 1]
    
    return prediction[0], risk_proba[0]