import joblib
import os
import sys

# Set path to models
# Set path to models
models_path = 'models'
scaler_path = os.path.join(models_path, 'scaler.pkl')
clustering_path = os.path.join(models_path, 'modelo_clustering.pkl')
stress_path = os.path.join(models_path, 'modelo_stress.pkl')

def inspect_object(name, path):
    try:
        print(f"\n--- Inspecting {name} ---")
        obj = joblib.load(path)
        print(f"Type: {type(obj)}")
        if hasattr(obj, 'feature_names_in_'):
            print(f"Features: {list(obj.feature_names_in_)}")
        if hasattr(obj, 'feature_importances_'):
            print("Has feature_importances_")
        if hasattr(obj, 'steps'): # Pipeline
            print("Pipeline steps:")
            for step_name, step_obj in obj.steps:
                print(f"  - {step_name}: {type(step_obj)}")
                if step_name == 'pca':
                    print("    Has PCA step")
                if hasattr(step_obj, 'feature_importances_'):
                    print(f"    Step {step_name} has feature_importances_")
        if hasattr(obj, 'n_clusters'):
            print(f"n_clusters: {obj.n_clusters}")
    except Exception as e:
        print(f"Error loading {name}: {e}")

inspect_object('Scaler', scaler_path)
inspect_object('Clustering', clustering_path)
inspect_object('Stress Model', stress_path)
