import joblib
import os
import numpy as np

MODELS_PATH = "/models"

try:
    scaler = joblib.load(os.path.join(MODELS_PATH, 'scaler.pkl'))
    print(f"Type of scaler: {type(scaler)}")
    
    if hasattr(scaler, 'scale_'):
        print("Scaler scales (std dev):")
        for i, (name, scale) in enumerate(zip(scaler.feature_names_in_, scaler.scale_)):
            print(f"{i}: {name} = {scale}")
            
    if hasattr(scaler, 'mean_'):
        print("\nScaler means:")
        for i, (name, mean) in enumerate(zip(scaler.feature_names_in_, scaler.mean_)):
            print(f"{i}: {name} = {mean}")

except Exception as e:
    print(f"Error inspecting scaler: {e}")
