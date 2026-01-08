import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Paths
DATA_PATH = 'data/raw_input.csv'
MODELS_PATH = 'models'
SCALER_PATH = os.path.join(MODELS_PATH, 'scaler.pkl')
MODEL_OUT_PATH = os.path.join(MODELS_PATH, 'modelo_clustering.pkl')

def retrain():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing to match the features expected by the models
    # Features from inspection: 
    # ['age', 'work_screen_hours', 'leisure_screen_hours', 'sleep_hours', 
    #  'productivity_0_100', 'exercise_minutes_per_week', 'social_hours_per_week', 
    #  'other_screen_hours', 'gender_Male', 'gender_Non-binary/Other', 
    #  'occupation_Retired', 'occupation_Self-employed', 'occupation_Student', 
    #  'occupation_Unemployed', 'work_mode_In-person', 'work_mode_Remote']
    
    # 1. One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=['gender', 'occupation', 'work_mode'], drop_first=False)
    
    # 2. Ensure all columns exist
    expected_features = [
        'age', 'work_screen_hours', 'leisure_screen_hours', 'sleep_hours', 
        'productivity_0_100', 'exercise_minutes_per_week', 'social_hours_per_week', 
        'other_screen_hours', 'gender_Male', 'gender_Non-binary/Other', 
        'occupation_Retired', 'occupation_Self-employed', 'occupation_Student', 
        'occupation_Unemployed', 'work_mode_In-person', 'work_mode_Remote'
    ]
    
    # Rename columns if necessary (e.g. if raw csv has different names)
    # Assuming raw csv matches the names used in app.py logic
    # Let's check raw_input.csv columns first? 
    # I'll assume they match for now, but add a check.
    
    # Add missing columns with 0
    for col in expected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Select features in correct order
    X = df_encoded[expected_features]
    
    # 3. Scale
    # We should use the EXISTING scaler to ensure compatibility with the Stress Model
    # OR refit a new scaler if we are sure the data is the same.
    # Safest is to load existing scaler.
    print("Loading scaler...")
    scaler = joblib.load(SCALER_PATH)
    
    X_scaled = scaler.transform(X)
    
    # 4. Train KMeans
    print("Training KMeans with k=3...")
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # 5. Train PCA
    from sklearn.decomposition import PCA
    print("Training PCA...")
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    
    # 6. Save
    print(f"Saving model to {MODEL_OUT_PATH}...")
    joblib.dump(kmeans, MODEL_OUT_PATH)
    
    PCA_OUT_PATH = os.path.join(MODELS_PATH, 'pca.pkl')
    print(f"Saving PCA to {PCA_OUT_PATH}...")
    joblib.dump(pca, PCA_OUT_PATH)
    
    print("Done.")

if __name__ == "__main__":
    retrain()
