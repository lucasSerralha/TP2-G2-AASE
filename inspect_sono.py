import joblib
import os
import pandas as pd
import numpy as np

MODELS_PATH = "/models"

try:
    model_sono = joblib.load(os.path.join(MODELS_PATH, 'modelo_sono.pkl'))
    print(f"Type of model_sono: {type(model_sono)}")
    
    if hasattr(model_sono, 'feature_names_in_'):
        print("Feature names in model_sono:")
        for i, name in enumerate(model_sono.feature_names_in_):
            print(f"{i}: {name}")
    else:
        print("model_sono does not have feature_names_in_")

    if hasattr(model_sono, 'coef_'):
        print(f"Coefficients: {model_sono.coef_}")
        print(f"Intercept: {model_sono.intercept_}")

except Exception as e:
    print(f"Error inspecting model: {e}")
