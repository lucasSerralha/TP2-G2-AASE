import joblib
import os
import sys

# Set path to models
models_path = '/models'
scaler_path = os.path.join(models_path, 'scaler.pkl')

try:
    print(f"Loading scaler from {scaler_path}...")
    scaler = joblib.load(scaler_path)
    
    if hasattr(scaler, 'feature_names_in_'):
        print("Feature names found in scaler:")
        for i, name in enumerate(scaler.feature_names_in_):
            print(f"{i}: {name}")
    else:
        print("Scaler does not have 'feature_names_in_' attribute.")
        # Try to infer from n_features_in_
        if hasattr(scaler, 'n_features_in_'):
             print(f"Scaler expects {scaler.n_features_in_} features.")

except Exception as e:
    print(f"Error loading scaler: {e}")
