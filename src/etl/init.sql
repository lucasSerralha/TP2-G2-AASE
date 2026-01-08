-- Drop tables if they exist
DROP TABLE IF EXISTS fact_mental_health;
DROP TABLE IF EXISTS dim_user;
DROP TABLE IF EXISTS dim_occupation;
DROP TABLE IF EXISTS dim_work_mode;

-- Create Dimension Tables
CREATE TABLE dim_user (
    user_id VARCHAR(50) PRIMARY KEY,
    age INT,
    gender VARCHAR(50)
);

CREATE TABLE dim_occupation (
    occupation_id SERIAL PRIMARY KEY,
    occupation_name VARCHAR(100) UNIQUE
);

CREATE TABLE dim_work_mode (
    work_mode_id SERIAL PRIMARY KEY,
    work_mode_name VARCHAR(50) UNIQUE
);

-- Create Fact Table
CREATE TABLE fact_mental_health (
    fact_id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) REFERENCES dim_user(user_id),
    occupation_id INT REFERENCES dim_occupation(occupation_id),
    work_mode_id INT REFERENCES dim_work_mode(work_mode_id),
    screen_time_hours FLOAT,
    work_screen_hours FLOAT,
    leisure_screen_hours FLOAT,
    sleep_hours FLOAT,
    sleep_quality_1_5 INT,
    stress_level_0_10 FLOAT,
    productivity_0_100 FLOAT,
    exercise_minutes_per_week INT,
    social_hours_per_week FLOAT,
    mental_wellness_index_0_100 FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
