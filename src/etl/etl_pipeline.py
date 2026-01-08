import os
import time
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
DATA_PATH = "/data/raw_input.csv"
MODELS_PATH = "/models"

def wait_for_db(engine):
    """Wait for the database to be ready."""
    while True:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("Database is ready!")
            break
        except OperationalError:
            print("Database not ready, waiting...")
            time.sleep(2)

def init_schema(engine):
    """Initialize the database schema."""
    # This is a simplified schema initialization. 
    # In a real scenario, you might want to use more specific types or a migration tool.
    # For this demo, we'll let pandas create the table, but we could define it here if needed.
    pass

def load_models():
    """Load machine learning models."""
    models = {}
    try:
        models['stress'] = joblib.load(os.path.join(MODELS_PATH, 'modelo_stress.pkl'))
        models['clustering'] = joblib.load(os.path.join(MODELS_PATH, 'modelo_clustering.pkl'))
        models['sono'] = joblib.load(os.path.join(MODELS_PATH, 'modelo_sono.pkl'))
        models['classificacao_stress'] = joblib.load(os.path.join(MODELS_PATH, 'modelo_classificacao_stress.pkl'))
        models['scaler'] = joblib.load(os.path.join(MODELS_PATH, 'scaler.pkl'))
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
    return models

def process_data(df, models):
    """Process data and run inference."""

    print("Processing data...")
    return df

def main():
    print("Starting ETL Job...")
    
    engine = create_engine(DATABASE_URL)
    wait_for_db(engine)
    
    # Extract
    print(f"Reading data from {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"File not found: {DATA_PATH}")
        return

    # Transform & Load Models
    models = load_models()
    
    # Load
    print("Loading data into PostgreSQL...")
    table_name = "mental_health_data"
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"Data loaded into table '{table_name}'.")

if __name__ == "__main__":
    main()
