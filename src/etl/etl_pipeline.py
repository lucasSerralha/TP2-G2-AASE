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
INIT_SQL_PATH = "init.sql" # Path relative to where the script is run (or absolute)

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
    """Initialize the database schema using init.sql."""
    print("Initializing schema...")
    try:
        with open(INIT_SQL_PATH, 'r') as f:
            sql_script = f.read()
        
        with engine.connect() as conn:
            # Split by ; to execute multiple statements if needed, 
            # but sqlalchemy execute(text()) might handle it depending on driver.
            # For safety with simple scripts:
            conn.execute(text(sql_script))
            conn.commit()
        print("Schema initialized successfully.")
    except Exception as e:
        print(f"Error initializing schema: {e}")

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

def main():
    print("Starting ETL Job...")
    
    engine = create_engine(DATABASE_URL)
    wait_for_db(engine)
    
    # Initialize Schema
    init_schema(engine)
    
    # Extract
    print(f"Reading data from {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"File not found: {DATA_PATH}")
        return

    # Transform & Load
    print("Loading data into Star Schema...")
    
    with engine.connect() as conn:
        # 1. Load Dimensions
        
        # dim_occupation
        occupations = df['occupation'].unique()
        df_occ = pd.DataFrame({'occupation_name': occupations})
        df_occ.to_sql('dim_occupation', engine, if_exists='append', index=False)
        print("Loaded dim_occupation")
        
        # dim_work_mode
        work_modes = df['work_mode'].unique()
        df_wm = pd.DataFrame({'work_mode_name': work_modes})
        df_wm.to_sql('dim_work_mode', engine, if_exists='append', index=False)
        print("Loaded dim_work_mode")
        
        # dim_user
        # Assuming user_id is unique in CSV for this demo
        df_user = df[['user_id', 'age', 'gender']].drop_duplicates()
        df_user.to_sql('dim_user', engine, if_exists='append', index=False)
        print("Loaded dim_user")
        
        # 2. Map IDs for Fact Table
        
        # Get IDs back from DB to ensure correct mapping
        occ_map = pd.read_sql("SELECT occupation_id, occupation_name FROM dim_occupation", conn)
        wm_map = pd.read_sql("SELECT work_mode_id, work_mode_name FROM dim_work_mode", conn)
        
        # Merge to get IDs
        df_fact = df.merge(occ_map, left_on='occupation', right_on='occupation_name')
        df_fact = df_fact.merge(wm_map, left_on='work_mode', right_on='work_mode_name')
        
        # Select columns for Fact Table
        fact_columns = [
            'user_id', 'occupation_id', 'work_mode_id',
            'screen_time_hours', 'work_screen_hours', 'leisure_screen_hours',
            'sleep_hours', 'sleep_quality_1_5', 'stress_level_0_10',
            'productivity_0_100', 'exercise_minutes_per_week',
            'social_hours_per_week', 'mental_wellness_index_0_100'
        ]
        
        df_fact_final = df_fact[fact_columns]
        
        # Load Fact Table
        df_fact_final.to_sql('fact_mental_health', engine, if_exists='append', index=False)
        print(f"Loaded {len(df_fact_final)} rows into fact_mental_health")

if __name__ == "__main__":
    main()
