import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os
import time
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import uuid
from datetime import datetime
from sklearn.decomposition import PCA

# --- CONFIGURAÇÃO ---
DATABASE_URL = os.getenv("DATABASE_URL")
MODELS_PATH = "/models"
PATIENTS_FILE = "/data/patients.json"

st.set_page_config(
    page_title="Mental Health Analytics Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GERENCIAMENTO DE TEMA ---
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'Light'

def toggle_theme():
    if st.session_state['theme'] == 'Light':
        st.session_state['theme'] = 'Dark'
    else:
        st.session_state['theme'] = 'Light'

# --- CSS PERSONALIZADO (Dinâmico) ---
LIGHT_THEME_CSS = """
    <style>
    /* Global */
    .stApp {
        background-color: #F8FAFC;
    }
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1E293B;
    }
    /* Cards */
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #E2E8F0;
    }
    .stMetric label { color: #64748B !important; }
    .stMetric [data-testid="stMetricValue"] { color: #1E293B !important; }
    
    /* Headers */
    h1, h2, h3 { color: #1E293B; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #F1F5F9; }
    
    /* Buttons */
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 12px;
        border: none;
    }
    </style>
"""

DARK_THEME_CSS = """
    <style>
    /* Global */
    .stApp {
        background-color: #0F172A;
    }
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #F8FAFC;
    }
    
    /* Force White Text on All Widgets */
    .stTextInput, .stNumberInput, .stSelectbox, .stDateInput, .stTimeInput, .stTextArea {
        color: #F8FAFC !important;
    }
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input {
        color: #F8FAFC !important;
        background-color: #1E293B !important;
    }
    .stSelectbox > div > div > div {
        color: #F8FAFC !important;
        background-color: #1E293B !important;
    }
    /* Dataframe/Table Text */
    [data-testid="stDataFrame"] {
        color: #F8FAFC !important;
    }
    [data-testid="stDataFrame"] div, [data-testid="stDataFrame"] span {
        color: #F8FAFC !important;
    }
    
    /* Cards */
    .stMetric {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5), 0 4px 6px -2px rgba(0, 0, 0, 0.3); /* Stronger Shadow */
        border: 1px solid rgba(255, 255, 255, 0.1); /* Light Stroke Highlight */
    }
    .stMetric label { color: #94A3B8 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #F8FAFC !important; }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 { color: #F8FAFC !important; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid rgba(255,255,255,0.1); /* Light Stroke */
        box-shadow: 4px 0 15px rgba(0,0,0,0.3); /* Shadow */
    }
    /* Sidebar Text & Inputs */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p {
        color: #F8FAFC !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0EA5E9 0%, #2563EB 100%);
        color: white;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    </style>
"""

# --- GERENCIAMENTO DE ESTADO E DADOS ---

# --- GERENCIAMENTO DE ESTADO E DADOS ---

class PatientManager:
    def __init__(self, filepath):
        self.filepath = filepath
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump([], f)

    def load_patients(self):
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except:
            return []

    def save_patient(self, patient_data):
        patients = self.load_patients()
        if 'id' not in patient_data:
            patient_data['id'] = str(uuid.uuid4())
            patient_data['created_at'] = datetime.now().isoformat()
            patient_data['history'] = []
            patients.append(patient_data)
        else:
            for i, p in enumerate(patients):
                if p['id'] == patient_data['id']:
                    patient_data['history'] = p.get('history', [])
                    patients[i] = patient_data
                    break
        
        with open(self.filepath, 'w') as f:
            json.dump(patients, f, indent=4)
        return patient_data

    def add_simulation_result(self, patient_id, result):
        patients = self.load_patients()
        for p in patients:
            if p['id'] == patient_id:
                if 'history' not in p:
                    p['history'] = []
                result['timestamp'] = datetime.now().isoformat()
                p['history'].append(result)
                break
        with open(self.filepath, 'w') as f:
            json.dump(patients, f, indent=4)

# --- FUNÇÕES DE CARREGAMENTO ---
@st.cache_resource
def get_database_connection():
    if not DATABASE_URL:
        return None
    return create_engine(DATABASE_URL)

@st.cache_resource
def load_models():
    models = {}
    required_files = ['modelo_stress.pkl', 'modelo_clustering.pkl', 'modelo_sono.pkl', 'scaler.pkl', 'pca.pkl']
    
    if not os.path.exists(os.path.join(MODELS_PATH, 'modelo_stress.pkl')):
        return None 

    try:
        for f in required_files:
            models[f.replace('modelo_', '').replace('.pkl', '')] = joblib.load(os.path.join(MODELS_PATH, f))
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {e}")
        return None
    return models



def load_data():
    engine = get_database_connection()
    df_db = pd.DataFrame()
    if engine:
        try:
            # Query Star Schema and reconstruct flat view
            query = """
            SELECT 
                f.*,
                u.age, u.gender,
                o.occupation_name as occupation,
                w.work_mode_name as work_mode
            FROM fact_mental_health f
            JOIN dim_user u ON f.user_id = u.user_id
            JOIN dim_occupation o ON f.occupation_id = o.occupation_id
            JOIN dim_work_mode w ON f.work_mode_id = w.work_mode_id
            """
            df_db = pd.read_sql(query, engine)
            
            # Drop ID columns not needed for analysis if desired, or keep them.
            # The app expects 'occupation' and 'work_mode' strings, which we aliased above.
            # We might want to drop the foreign keys to clean up, but it's not strictly necessary.
        except Exception as e:
            st.error(f"Erro ao carregar dados do banco: {e}")
            pass
    
    pm = PatientManager(PATIENTS_FILE)
    patients = pm.load_patients()
    
    patient_records = []
    for p in patients:
        for sim in p.get('history', []):
            record = sim.copy()
            record['source'] = 'patient_simulation'
            record['patient_id'] = p['id']
            record['age'] = p.get('age')
            record['gender'] = p.get('gender')
            record['occupation'] = p.get('occupation')
            patient_records.append(record)
            
    df_patients = pd.DataFrame(patient_records)
    
    if not df_db.empty and not df_patients.empty:
        return pd.concat([df_db, df_patients], ignore_index=True)
    elif not df_db.empty:
        return df_db
    elif not df_patients.empty:
        return df_patients
    else:
        return pd.DataFrame()

# --- AUTENTICAÇÃO ---
# --- AUTENTICAÇÃO REMOVIDA ---
# def check_password(): ...


# --- COMPONENTES VISUAIS ---

def plot_gauge(value, title, min_val=0, max_val=10, key=None):
    is_dark = st.session_state['theme'] == 'Dark'
    text_color = '#F8FAFC' if is_dark else '#1E293B'
    bg_color = '#1E293B' if is_dark else 'white'
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'color': text_color, 'size': 20}},
        number = {'font': {'color': text_color}},
        gauge = {
            'axis': {'range': [min_val, max_val], 'tickcolor': text_color},
            'bar': {'color': text_color}, 
            'steps': [
                {'range': [0, 4], 'color': "#10B981"}, # Emerald
                {'range': [4, 7], 'color': "#F59E0B"}, # Amber
                {'range': [7, 10], 'color': "#EF4444"} # Red
            ],
            'threshold': {
                'line': {'color': "white" if is_dark else "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            },
            'bgcolor': bg_color
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), 
                      paper_bgcolor="rgba(0,0,0,0)", 
                      font={'family': "Inter, sans-serif", 'color': text_color})
    st.plotly_chart(fig, use_container_width=True, key=key)

# --- PÁGINAS ---

def render_dashboard(df):
    st.title("📊 Dashboard Analytics")
    
    if df.empty:
        st.warning("Aguardando dados...")
        return

    # --- FILTROS LATERAIS (GLOBAL) ---
    st.sidebar.header("Filtros de População")
    
    # Defaults
    all_genders = df['gender'].unique() if 'gender' in df.columns else []
    all_occs = df['occupation'].unique() if 'occupation' in df.columns else []
    all_modes = df['work_mode'].unique() if 'work_mode' in df.columns else []
    
    sel_gender = st.sidebar.multiselect("Gênero", all_genders, default=all_genders)
    sel_occ = st.sidebar.multiselect("Ocupação", all_occs, default=all_occs)
    sel_mode = st.sidebar.multiselect("Modo de Trabalho", all_modes, default=all_modes)
    
    # Apply Filters
    mask = pd.Series(True, index=df.index)
    if sel_gender: mask &= df['gender'].isin(sel_gender)
    if sel_occ: mask &= df['occupation'].isin(sel_occ)
    if sel_mode: mask &= df['work_mode'].isin(sel_mode)
    
    df_filtered = df[mask]
    
    # --- LINHA 1: KPIS ---
    st.markdown("### Métricas Globais")
    k1, k2, k3 = st.columns(3)
    
    avg_stress = df_filtered['stress_level_0_10'].mean() if 'stress_level_0_10' in df_filtered.columns else 0
    avg_sleep = df_filtered['sleep_hours'].mean() if 'sleep_hours' in df_filtered.columns else 0
    avg_well = df_filtered['mental_wellness_index_0_100'].mean() if 'mental_wellness_index_0_100' in df_filtered.columns else 0
    
    k1.metric("Média de Estresse (0-10)", f"{avg_stress:.2f}", delta="Alto Risco" if avg_stress > 8 else "Normal", delta_color="inverse")
    k2.metric("Média de Sono (Horas)", f"{avg_sleep:.1f}h")
    k3.metric("Índice de Bem-Estar", f"{avg_well:.1f}%")
    
    st.divider()
    
    # --- LINHA 2: FATOR DIGITAL ---
    st.markdown("### O Fator Digital vs. Saúde Mental")
    c1, c2 = st.columns([6, 6]) # 12-col grid: 6+6
    
    is_dark = st.session_state['theme'] == 'Dark'
    plotly_template = "plotly_dark" if is_dark else "plotly_white"
    scatter_color = "#38BDF8" if is_dark else "#2563EB"
    
    with c1:
        if 'screen_time_hours' in df_filtered.columns and 'stress_level_0_10' in df_filtered.columns:
            fig = px.scatter(df_filtered, x='screen_time_hours', y='stress_level_0_10', 
                             trendline="ols", title="Tempo de Ecrã vs. Estresse",
                             opacity=0.7, 
                             color_discrete_sequence=[scatter_color]) 
            fig.update_traces(marker=dict(size=8, line=dict(width=1, color='#0F172A' if is_dark else 'white')))
            fig.update_layout(
                template=plotly_template, 
                font={'family': "Inter, sans-serif", 'color': 'white' if is_dark else '#1E293B'}, 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                title_font_color="white" if is_dark else "#1E293B",
                legend_font_color="white" if is_dark else "#1E293B"
            )

            st.plotly_chart(fig, use_container_width=True, key="scatter_screen_stress")
            
    with c2:
        if 'occupation' in df_filtered.columns and 'stress_level_0_10' in df_filtered.columns:
            fig = px.box(df_filtered, x='occupation', y='stress_level_0_10', color='occupation',
                         title="Distribuição de Estresse por Ocupação",
                         color_discrete_sequence=px.colors.qualitative.Bold if is_dark else px.colors.qualitative.Prism) 
            fig.update_layout(
                template=plotly_template, 
                font={'family': "Inter, sans-serif", 'color': 'white' if is_dark else '#1E293B'}, 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                title_font_color="white" if is_dark else "#1E293B",
                legend_font_color="white" if is_dark else "#1E293B"
            )

            st.plotly_chart(fig, use_container_width=True, key="box_occupation_stress")

    # --- LINHA 3: SEGMENTAÇÃO (PCA) ---
    st.markdown("### Segmentação e Perfis (PCA Clusters)")
    
    models = load_models()
    if models and 'scaler' in models and 'clustering' in models:
        try:
            # Prepare data for PCA (must match training features)
            scaler = models['scaler']
            kmeans = models['clustering']
            
            # Filter numeric columns used in training
            features = list(scaler.feature_names_in_)
            
            # We need to ensure df_filtered has all necessary columns, including one-hot encoded ones
            # This is tricky with raw data. For visualization, we might need to re-process or use a subset.
            # Simplified approach: Try to construct features from available data
            
            # Check if we have pre-processed features or need to generate them on the fly
            # For this demo, we'll assume df_filtered has the raw columns and we need to encode/scale
            
            # Helper to encode (simplified, assumes same categories as training)
            # In a real app, use the same pipeline used for training
            
            # If we can't perfectly reconstruct the training set, we skip PCA or show a placeholder
            # But let's try to map what we can
            
            # Create a temporary DF with required columns, filling missing with 0
            df_pca_input = pd.DataFrame(0, index=df_filtered.index, columns=features)
            
            # Map numeric
            numeric_map = {
                'age': 'age', 'work_screen_hours': 'work_screen_hours', 
                'leisure_screen_hours': 'leisure_screen_hours', 'sleep_hours': 'sleep_hours',
                'productivity_0_100': 'productivity_0_100', 'exercise_minutes_per_week': 'exercise_minutes_per_week',
                'social_hours_per_week': 'social_hours_per_week', 'other_screen_hours': 'other_screen_hours'
            }
            for feat, col in numeric_map.items():
                if col in df_filtered.columns:
                    df_pca_input[feat] = df_filtered[col]
            
            # Map categorical (One-Hot)
            for col in ['gender', 'occupation', 'work_mode']:
                if col in df_filtered.columns:
                    for val in df_filtered[col].unique():
                        feat_name = f"{col}_{val}"
                        if feat_name in features:
                            df_pca_input.loc[df_filtered[col] == val, feat_name] = 1
            
            # Handle NaNs (fill with 0 or mean)
            df_pca_input = df_pca_input.fillna(0)

            # Scale
            X_scaled = scaler.transform(df_pca_input)
            
            # Predict Clusters
            clusters = kmeans.predict(X_scaled)
            
            # PCA Transform (Use pre-fitted model)
            if 'pca' in models:
                pca = models['pca']
                components = pca.transform(X_scaled)
            else:
                # Fallback if PCA model not loaded (should not happen if load_models updated)
                pca = PCA(n_components=2)
                components = pca.fit_transform(X_scaled)
            
            df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
            df_pca['Cluster'] = clusters.astype(str)

            # Define Colors for Clusters
            if is_dark:
                color_map = {'0': '#94A3B8', '1': '#22D3EE', '2': '#A78BFA'}
                template = "plotly_dark"
            else:
                color_map = {'0': '#475569', '1': '#06B6D4', '2': '#6366F1'}
                template = "plotly_white"
            
            fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster', 
                                 title="Mapa de Perfis (K-Means Clustering)",
                                 template=template,
                                 color_discrete_map=color_map,
                                 opacity=0.9,
                                 width=800, height=500)
            fig_pca.update_traces(marker=dict(size=10, line=dict(width=1, color='#0F172A' if is_dark else 'white')))
            fig_pca.update_traces(marker=dict(size=10, line=dict(width=1, color='#0F172A' if is_dark else 'white')))
            fig_pca.update_layout(
                font={'family': "Inter, sans-serif", 'color': 'white' if is_dark else '#1E293B'}, 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                title_font_color="white" if is_dark else "#1E293B",
                legend_font_color="white" if is_dark else "#1E293B"
            )

            st.plotly_chart(fig_pca, use_container_width=True, key="pca_cluster_map")
            
        except Exception as e:
            st.error(f"Erro ao gerar mapa de clusters: {e}")
    else:
        st.info("Modelos de clustering não disponíveis.")

    # --- EXTRAS ---
    c3, c4 = st.columns(2)
    
    is_dark = st.session_state['theme'] == 'Dark'
    plotly_template = "plotly_dark" if is_dark else "plotly_white"
    
    with c3:
        if 'productivity_0_100' in df_filtered.columns and 'stress_level_0_10' in df_filtered.columns:
             fig = px.density_heatmap(df_filtered, x='productivity_0_100', y='stress_level_0_10', 
                                      title="Produtividade vs Estresse",
                                      color_continuous_scale="Blues" if not is_dark else "Viridis")
             fig.update_layout(
                template=plotly_template, 
                font={'family': "Inter, sans-serif", 'color': 'white' if is_dark else '#1E293B'}, 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                title_font_color="white" if is_dark else "#1E293B"
             )

             st.plotly_chart(fig, use_container_width=True, key="heatmap_prod_stress")
    with c4:
        if 'exercise_minutes_per_week' in df_filtered.columns:
            fig = px.histogram(df_filtered, x='exercise_minutes_per_week', nbins=20,
                               title="Distribuição de Exercício Físico",
                               color_discrete_sequence=["#2563EB" if not is_dark else "#38BDF8"])
            fig.update_layout(
                template=plotly_template, 
                font={'family': "Inter, sans-serif", 'color': 'white' if is_dark else '#1E293B'}, 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                title_font_color="white" if is_dark else "#1E293B"
            )

            st.plotly_chart(fig, use_container_width=True, key="hist_exercise")


def render_playground(df_avg):
    st.title("🤖 AI Simulator (Simulador Clínico)")
    
    models = load_models()
    
    # --- BLOCO A: PERFIL ---
    st.sidebar.header("Perfil do Paciente")
    
    patient = st.session_state.get('selected_patient')
    if patient:
        st.info(f"Simulando para: **{patient['name']}**")
        def_age = int(patient['age'])
        def_gender = patient['gender']
        def_occ = patient['occupation']
    else:
        st.sidebar.info("Modo Genérico (Selecione um paciente na aba Gestão para salvar)")
        def_age = 30
        def_gender = "Male"
        def_occ = "Employed"

    # Inputs Demográficos (Sidebar ou Topo)
    c1, c2, c3, c4 = st.columns(4)
    age = c1.number_input("Idade", 18, 90, def_age)
    gender = c2.selectbox("Gênero", ["Male", "Female", "Non-binary/Other"], index=["Male", "Female", "Non-binary/Other"].index(def_gender))
    occupation = c3.selectbox("Ocupação", ["Employed", "Self-employed", "Student", "Retired", "Unemployed"], index=["Employed", "Self-employed", "Student", "Retired", "Unemployed"].index(def_occ))
    work_mode = c4.selectbox("Modo de Trabalho", ["Remote", "Hybrid", "In-person"])
    
    st.divider()
    
    # --- BLOCO B: VARIÁVEIS COMPORTAMENTAIS ---
    st.subheader("Variáveis Comportamentais (Cenários 'E se?')")
    
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        screen_time = st.slider("Tempo Total de Ecrã (h)", 0.0, 24.0, 8.0)
        work_screen = st.slider("...dos quais Trabalho (h)", 0.0, 24.0, 6.0)
        leisure_screen = st.slider("...dos quais Lazer (h)", 0.0, 24.0, 2.0)
        sleep_hours = st.slider("Horas de Sono", 0.0, 12.0, 7.0)
        exercise = st.slider("Exercício (min/semana)", 0, 1000, 150)
        social = st.slider("Interação Social (h/semana)", 0.0, 50.0, 10.0)
        productivity = st.slider("Produtividade Percebida (0-10)", 0, 10, 7)
        
        run_sim = st.button("Rodar Simulação", type="primary")

    # --- BLOCO C: RESULTADOS ---
    with col_output:
        if run_sim:
            # Prepare Input
            input_data = pd.DataFrame({
                'age': [age],
                'work_screen_hours': [work_screen],
                'leisure_screen_hours': [leisure_screen],
                'sleep_hours': [sleep_hours],
                'productivity_0_100': [productivity * 10],
                'exercise_minutes_per_week': [exercise],
                'social_hours_per_week': [social],
                'other_screen_hours': [0.0], # Dummy
                # One-Hot Encoding (Manual for demo)
                'gender_Male': [1 if gender == 'Male' else 0],
                'gender_Non-binary/Other': [1 if gender == 'Non-binary/Other' else 0],
                'occupation_Retired': [1 if occupation == 'Retired' else 0],
                'occupation_Self-employed': [1 if occupation == 'Self-employed' else 0],
                'occupation_Student': [1 if occupation == 'Student' else 0],
                'occupation_Unemployed': [1 if occupation == 'Unemployed' else 0],
                'work_mode_In-person': [1 if work_mode == 'In-person' else 0],
                'work_mode_Remote': [1 if work_mode == 'Remote' else 0]
            })
            
            if models and 'stress' in models:
                try:
                    scaler = models['scaler']
                    model_stress = models['stress']
                    model_sleep = models['sono']
                    
                    X_scaled = scaler.transform(input_data)
                    stress_pred = model_stress.predict(X_scaled)[0]
                    sleep_pred = model_sleep.predict(X_scaled)[0]
                    
                    # 1. Gauge Chart Stresse
                    plot_gauge(stress_pred, "Nível de Estresse Previsto", key="gauge_stress_sim")
                    
                    # 2. Card Sono
                    if sleep_pred > 4:
                        sleep_label = "Ótima"
                    elif sleep_pred > 3:
                        sleep_label = "Boa"
                    else:
                        sleep_label = "-Ruim" # Prefix with - to make it Red/Down arrow
                        
                    st.metric("Qualidade do Sono Prevista", f"{sleep_pred:.2f}/5", delta=sleep_label)
                    
                    # 3. XAI (Feature Importance)
                    st.markdown("#### 🧠 Clinical Insight (XAI)")
                    if hasattr(model_stress, 'feature_importances_'):
                        importances = model_stress.feature_importances_
                        feats = scaler.feature_names_in_
                        # Sort
                        indices = np.argsort(importances)[::-1]
                        top_3 = indices[:3]
                        
                        st.write("Fatores de maior peso para esta predição:")
                        for i in top_3:
                            st.write(f"- **{feats[i]}**: {importances[i]*100:.1f}% impacto")
                            
                    # 4. Recommendations
                    st.markdown("#### 💡 Recomendações Personalizadas")
                    recommendations = []
                    
                    if stress_pred > 5:
                        recommendations.append("⚠️ **Alto Risco de Estresse**: Considere pausas ativas a cada 50 minutos de trabalho.")
                        if work_screen > 6:
                            recommendations.append("📉 **Reduzir Tempo de Ecrã Profissional**: Tente delegar tarefas ou usar métodos offline quando possível.")
                    
                    if sleep_pred < 3.5:
                        recommendations.append("🌙 **Melhorar Sono**: Evite ecrãs 1h antes de dormir. A luz azul suprime a melatonina.")
                        if sleep_hours < 7:
                            recommendations.append("⏰ **Aumentar Horas de Sono**: Tente dormir pelo menos 7-8 horas por noite.")
                            
                    if exercise < 150:
                        recommendations.append("🏃 **Atividade Física**: Aumente para pelo menos 150 min/semana para reduzir o cortisol.")
                        
                    if social < 5:
                        recommendations.append("🗣️ **Socialização**: Aumentar o tempo com amigos/família pode melhorar o bem-estar mental.")
                        
                    if not recommendations:
                        st.success("🎉 Parabéns! Seus hábitos atuais indicam um perfil saudável.")
                    else:
                        for rec in recommendations:
                            st.info(rec)
                            
                    # Save Result
                    if patient:
                        pm = PatientManager(PATIENTS_FILE)
                        result = {
                            'stress_level_0_10': float(stress_pred),
                            'sleep_quality_1_5': float(sleep_pred),
                            'screen_time_hours': float(screen_time),
                            'timestamp': datetime.now().isoformat()
                        }
                        pm.add_simulation_result(patient['id'], result)
                        st.success("Resultado salvo no histórico!")
                        
                except Exception as e:
                    st.error(f"Erro na predição: {e}")
            else:
                st.warning("Modelos não carregados.")

def render_patient_management():
    st.title("👥 Gestão de Pacientes")
    
    pm = PatientManager(PATIENTS_FILE)
    patients = pm.load_patients()
    
    # Add Patient
    with st.expander("➕ Adicionar Novo Paciente"):
        with st.form("new_patient"):
            c1, c2 = st.columns(2)
            name = c1.text_input("Nome")
            age = c2.number_input("Idade", 18, 100, 30)
            c3, c4 = st.columns(2)
            gender = c3.selectbox("Gênero", ["Male", "Female", "Non-binary/Other"])
            occupation = c4.selectbox("Ocupação", ["Employed", "Student", "Self-employed", "Retired", "Unemployed"])
            
            if st.form_submit_button("Salvar"):
                pm.save_patient({"name": name, "age": age, "gender": gender, "occupation": occupation})
                st.success("Paciente adicionado!")
                st.rerun()
                
    # Table
    if patients:
        data = []
        for p in patients:
            last_stress = "N/A"
            last_date = "N/A"
            if p.get('history'):
                last = p['history'][-1]
                last_stress = f"{last.get('stress_level_0_10', 0):.2f}"
                last_date = last.get('timestamp', 'N/A')[:10]
                
            data.append({
                "ID": p['id'][:8],
                "Nome": p['name'],
                "Idade": p['age'],
                "Ocupação": p['occupation'],
                "Último Estresse": last_stress,
                "Última Análise": last_date
            })
            
        df_patients = pd.DataFrame(data)
        st.dataframe(df_patients, use_container_width=True)
        
        st.divider()
        
        # Selection for Actions
        patient_names = [p['name'] for p in patients]
        selected_name = st.selectbox("Selecione um Paciente para ver Histórico / Simular", [""] + patient_names)
        
        if selected_name:
            selected_patient = next(p for p in patients if p['name'] == selected_name)
            st.session_state['selected_patient'] = selected_patient
            
            c1, c2 = st.columns([2, 1])
            
            with c2:
                st.write("### Ações")
                if st.button("Ir para Simulador com este Paciente"):
                    st.session_state['page'] = "AI Simulator"
                    st.rerun()

            with c1:
                st.write(f"**Histórico de Estresse: {selected_patient['name']}**")
                if selected_patient.get('history'):
                    hist_df = pd.DataFrame(selected_patient['history'])
                    
                    is_dark = st.session_state['theme'] == 'Dark'
                    plotly_template = "plotly_dark" if is_dark else "plotly_white"
                    line_color = "#38BDF8" if is_dark else "#2563EB"
                    
                    fig = px.line(hist_df, x='timestamp', y='stress_level_0_10', markers=True,
                                  title="Evolução do Estresse")
                    fig.update_traces(line_color=line_color)
                    fig.update_layout(
                        template=plotly_template, 
                        font={'family': "Inter, sans-serif", 'color': 'white' if is_dark else '#1E293B'}, 
                        paper_bgcolor="rgba(0,0,0,0)", 
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=20, r=20, t=20, b=20),
                        height=300,
                        title_font_color="white" if is_dark else "#1E293B"
                    )

                    st.plotly_chart(fig, use_container_width=True, key="line_patient_history")
                else:
                    st.info("Sem histórico.")
    else:
        st.info("Nenhum paciente cadastrado.")

def main():
    # Navigation
    st.sidebar.title("MenteSã Pro")
    
    # Theme Toggle
    if 'theme' not in st.session_state:
        st.session_state['theme'] = 'Light'
        
    on = st.sidebar.toggle("Modo Escuro", value=(st.session_state['theme'] == 'Dark'))
    if on and st.session_state['theme'] != 'Dark':
        st.session_state['theme'] = 'Dark'
        st.rerun()
    elif not on and st.session_state['theme'] != 'Light':
        st.session_state['theme'] = 'Light'
        st.rerun()
        
    # Inject CSS
    if st.session_state['theme'] == 'Dark':
        st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    else:
        st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)
    
    if 'page' not in st.session_state:
        st.session_state['page'] = "Dashboard Analytics"
        
    options = ["Dashboard Analytics", "Gestão de Pacientes", "AI Simulator"]
    idx = options.index(st.session_state.get('page', "Dashboard Analytics"))
    
    selection = st.sidebar.radio("Menu", options, index=idx)
    
    if selection != st.session_state['page']:
        st.session_state['page'] = selection
        st.rerun()
        
    df = load_data()
    
    if st.session_state['page'] == "Dashboard Analytics":
        render_dashboard(df)
    elif st.session_state['page'] == "Gestão de Pacientes":
        render_patient_management()
    elif st.session_state['page'] == "AI Simulator":
        render_playground(df)

if __name__ == "__main__":
    main()