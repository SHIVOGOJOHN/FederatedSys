import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from pydantic import BaseModel
import numpy as np
import pandas as pd
from fastapi import FastAPI
import uvicorn

import os
from datetime import datetime
import shap

# Define the expected order of features for the FastAPI model
# This should match the 'expected_order' in your FastAPI app.py
expected_order = ['Age', 'Income', 'Loyalty_Score', 'Prior_Purchases',
    'Avg_Spend', 'Recency', 'Browsing_Time', 'Clicks_On_Promo',
    'Purchase_Frequency', 'High_Value_Customer', 'Engagement_Score',
    'Gender_Male', 'Promotion_Type_Discount',
    'Promotion_Type_FlashSale', 'Promotion_Type_LoyaltyPoints',
    'Channel_Email', 'Channel_In_store', 'Channel_SMS',
    'Time_of_Day_Evening', 'Time_of_Day_Morning']

app=FastAPI()

# Load models and scalers
model_path = "data/global_model_round_10.keras"
# Rebuild model architecture and load weights
fed_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(expected_order),)), # Use len(expected_order) for input shape
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
fed_model.load_weights(model_path)

fed_scaler = joblib.load("data/scaler_store_a.joblib")

# Define a background dataset for SHAP DeepExplainer
# Use real data from user predictions if available, otherwise fall back to random.
retraining_file_path = 'data/manual_predictions_for_retraining.csv'
try:
    df = pd.read_csv(retraining_file_path)
    if 'Will_Buy' in df.columns:
        df = df.drop(columns=['Will_Buy'])
    
    # Ensure column order is correct
    df = df[expected_order]

    sample_size = min(len(df), 100)
    background_df = df.sample(n=sample_size, random_state=1) # Use random_state for reproducibility
    
    # The background data must be scaled, just like the model's input.
    background_data = fed_scaler.transform(background_df.to_numpy())

except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
    # Fallback to random data if the file is not found, empty, or has the wrong columns.
    background_data_shape = (100, len(expected_order))
    background_data = fed_scaler.transform(np.random.rand(*background_data_shape))

#define pydantic scheme for the model

class FedInput(BaseModel):  
    Age                          : int
    Income                       : float
    Loyalty_Score                : int
    Prior_Purchases              : int
    Avg_Spend                    : float
    Recency                      : int
    Browsing_Time                : float
    Clicks_On_Promo              : int
    Purchase_Frequency           : float
    High_Value_Customer          : int
    Engagement_Score             : int 
    Gender_Male                  : int
    Promotion_Type_Discount      : int
    Promotion_Type_FlashSale     : int
    Promotion_Type_LoyaltyPoints : int
    Channel_Email                : int
    Channel_In_store             : int
    Channel_SMS                  : int
    Time_of_Day_Evening          : int
    Time_of_Day_Morning          : int

# Testing route
@app.get('/')
def ping():
    return {"message": "pong"}

# Route model prediction
@app.post('/fed_sys')
def predict_fed(input: FedInput):
    input_dict = input.model_dump()
    input_array = np.array([[input_dict[feature] for feature in expected_order]])
    input_scaled = fed_scaler.transform(input_array)
    probability = fed_model.predict(input_scaled)[0][0] # Access the single probability value
    
    # Apply a threshold to get a binary prediction
    binary_prediction = 1 if probability >= 0.5 else 0
    
    model_version = model_path.split('_')[-1].split('.')[0]
    last_updated = os.path.getmtime(model_path)

    # Calculate SHAP values
    explainer = shap.DeepExplainer(fed_model, background_data)
    shap_values = explainer.shap_values(input_scaled) # This is an array of shape (1, 20, 1)

    # Map SHAP values to feature names
    feature_importances = {}
    # shap_values[0] gives us the (20, 1) array for our single sample
    # shap_values[0][i] gives the i-th feature's value in a (1,) array
    # shap_values[0][i][0] gives the float
    for i, feature_name in enumerate(expected_order):
        feature_importances[feature_name] = float(shap_values[0][i][0])

    # Prepare data for SHAP plot
    shap_plot_data = {
        "shap_values": [v[0] for v in shap_values[0]], # Reshape from (20, 1) to (20,)
        "expected_value": float(explainer.expected_value[0]),
        "feature_values": input_array[0].tolist(),
        "feature_names": expected_order
    }

    return {
        "prediction": binary_prediction, 
        "probability": float(probability),
        "model_version": f"Round {model_version}",
        "last_updated": last_updated,
        "feature_importances": feature_importances, # Keep this for any other use
        "shap_plot_data": shap_plot_data # Add new data for plotting
    }

from sklearn.metrics import accuracy_score

@app.get('/dashboard_data')
def get_dashboard_data():
    retraining_file_path = 'data/manual_predictions_for_retraining.csv'
    try:
        df = pd.read_csv(retraining_file_path)
        if df.empty or 'Will_Buy' not in df.columns:
            return {"error": "No retraining data available yet."}

        # Ensure column order is correct and separate features (X) from target (y)
        X = df[expected_order]
        y_true = df['Will_Buy']

        # --- 1. Calculate Accuracy ---
        # Scale features before prediction
        X_scaled = fed_scaler.transform(X)
        y_pred_probs = fed_model.predict(X_scaled)
        y_pred = (y_pred_probs > 0.5).astype(int)
        accuracy = accuracy_score(y_true, y_pred)

        # --- 2. Calculate Global Feature Importances ---
        sample_size_shap = min(len(X_scaled), 200) # Use up to 200 samples for global explanation
        X_scaled_sample = X_scaled[np.random.choice(X_scaled.shape[0], sample_size_shap, replace=False)]

        explainer = shap.DeepExplainer(fed_model, background_data)
        shap_values_global = explainer.shap_values(X_scaled_sample)

        if isinstance(shap_values_global, list):
            shap_values_global_array = shap_values_global[0]
        else:
            shap_values_global_array = shap_values_global

        if len(shap_values_global_array.shape) == 3 and shap_values_global_array.shape[2] == 1:
            shap_values_global_array = shap_values_global_array.reshape(shap_values_global_array.shape[0], shap_values_global_array.shape[1])

        global_importances = np.abs(shap_values_global_array).mean(axis=0)
        global_feature_importances = {
            feature_name: float(global_importances[i])
            for i, feature_name in enumerate(expected_order)
        }

        # --- 3. Calculate PDP Data for the Most Important Feature ---
        pdp_data = {} # Initialize pdp_data to an empty dict
        if global_feature_importances: # Only calculate PDP if global importances are available
            most_important_feature = max(global_feature_importances, key=global_feature_importances.get)
            most_important_feature_index = expected_order.index(most_important_feature)

            X_mean_scaled = X_scaled.mean(axis=0)

            feature_min = X_scaled[:, most_important_feature_index].min()
            feature_max = X_scaled[:, most_important_feature_index].max()
            
            feature_values_for_pdp = np.linspace(feature_min, feature_max, 50)
            
            predictions_for_pdp = []

            for val in feature_values_for_pdp:
                synthetic_sample_scaled = np.copy(X_mean_scaled)
                synthetic_sample_scaled[most_important_feature_index] = val
                
                prob = fed_model.predict(synthetic_sample_scaled.reshape(1, -1))[0][0]
                predictions_for_pdp.append(float(prob))
            
            pdp_data = {
                "feature_name": most_important_feature,
                "feature_values": feature_values_for_pdp.tolist(),
                "predictions": predictions_for_pdp
            }

        return {"accuracy": accuracy, "global_feature_importances": global_feature_importances, "pdp_data": pdp_data}

    except FileNotFoundError:
        return {"error": "No retraining data available yet to generate a dashboard."}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}



