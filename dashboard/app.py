import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os
import time
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURAÇÃO ---
DATABASE_URL = os.getenv("DATABASE_URL")
MODELS_PATH = "/models"

st.set_page_config(
    page_title="Mental Health Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONALIZADO PARA ESTÉTICA ---
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNÇÕES DE CARREGAMENTO (CACHED) ---
@st.cache_resource
def get_database_connection():
    if not DATABASE_URL:
        return None
    return create_engine(DATABASE_URL)

@st.cache_resource
def load_models():
    models = {}
    required_files = ['modelo_stress.pkl', 'modelo_clustering.pkl', 'modelo_sono.pkl', 'scaler.pkl']
    
    # Mocking models for demonstration if files don't exist (Remove this in production)
    # This ensures the dashboard doesn't crash if you copy-paste this without the .pkl files
    if not os.path.exists(os.path.join(MODELS_PATH, 'modelo_stress.pkl')):
        return None 

    try:
        for f in required_files:
            models[f.replace('modelo_', '').replace('.pkl', '')] = joblib.load(os.path.join(MODELS_PATH, f))
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {e}")
        return None
    return models

@st.cache_data(ttl=600) # Cache data for 10 mins
def load_data():
    engine = get_database_connection()
    if not engine:
        # Retorna dados falsos para visualização se não houver DB conectado (Fallback)
        return pd.DataFrame() 
        
    try:
        query = "SELECT * FROM mental_health_data"
        df = pd.read_sql(query, engine)
        return df
    except Exception:
        return pd.DataFrame()

# --- COMPONENTES VISUAIS ---

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Mapa de Correlação")
    st.plotly_chart(fig, use_container_width=True)

def plot_radar_chart(user_input, avg_data):
    """Compara o usuário com a média da população"""
    categories = ['Screen Time', 'Sleep', 'Productivity', 'Social', 'Exercise']
    
    # Normalizando valores para escala 0-1 (aproximada para visualização)
    # No mundo real, usaríamos o MinMaxScaler carregado
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[user_input['screen_time'], user_input['sleep'], user_input['prod']/10, user_input['social'], user_input['exercise']*2],
        theta=categories,
        fill='toself',
        name='Você (Simulação)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[avg_data['screen_time'], avg_data['sleep'], avg_data['prod']/10, avg_data['social'], avg_data['exercise']*2],
        theta=categories,
        fill='toself',
        name='Média da População'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 12])),
        showlegend=True,
        title="Seus Hábitos vs. Média"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- PÁGINAS ---

def render_dashboard(df):
    st.header("📊 Panorama de Saúde Mental")
    
    if df.empty:
        st.warning("Aguardando dados... (Verifique a conexão com o banco)")
        return

    # 1. Filtros Globais (Sidebar)
    st.sidebar.markdown("### Filtros de Dados")
    
    # Get unique values for defaults
    all_genders = df['gender'].unique()
    all_work_modes = df['work_mode'].unique()
    
    gender_filter = st.sidebar.multiselect("Gênero", options=all_genders, default=all_genders)
    work_filter = st.sidebar.multiselect("Modo de Trabalho", options=all_work_modes, default=all_work_modes)
    
    # Logic: If filter is empty, treat as "Select All"
    if not gender_filter:
        gender_filter = all_genders
    if not work_filter:
        work_filter = all_work_modes
    
    df_filtered = df[df['gender'].isin(gender_filter) & df['work_mode'].isin(work_filter)]

    # 2. KPIs (Key Performance Indicators)
    st.markdown("### 📈 Indicadores Chave")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    # Correct column names based on CSV schema
    col_stress = 'stress_level_0_10'
    col_sleep = 'sleep_hours'
    col_wellness = 'mental_wellness_index_0_100'
    col_social = 'social_hours_per_week'
    col_screen = 'screen_time_hours'
    
    avg_stress = df_filtered[col_stress].mean() if col_stress in df_filtered.columns else 0
    avg_sleep = df_filtered[col_sleep].mean() if col_sleep in df_filtered.columns else 0
    avg_wellness = df_filtered[col_wellness].mean() if col_wellness in df_filtered.columns else 0
    
    kpi1.metric("Total de Registros", len(df_filtered))
    kpi2.metric("Nível Médio de Estresse", f"{avg_stress:.1f}/10", delta_color="inverse")
    kpi3.metric("Média Horas de Sono", f"{avg_sleep:.1f}h")
    kpi4.metric("Índice de Bem-Estar", f"{avg_wellness:.1f}/100" if col_wellness in df_filtered.columns else "N/A")

    st.markdown("---")

    # 3. Análises Detalhadas
    st.subheader("🔍 Análises Detalhadas de Saúde Mental")
    
    # Row 1: Stress Analysis
    st.markdown("#### 1. Análise de Estresse")
    r1_col1, r1_col2 = st.columns(2)
    
    with r1_col1:
        if col_stress in df_filtered.columns:
            fig_stress_occ = px.box(
                df_filtered, x='occupation', y=col_stress, color='occupation', 
                title="Estresse por Ocupação"
            )
            st.plotly_chart(fig_stress_occ, use_container_width=True)
            
    with r1_col2:
        if col_stress in df_filtered.columns:
            fig_stress_gen = px.violin(
                df_filtered, x='gender', y=col_stress, color='gender', box=True,
                title="Estresse por Gênero"
            )
            st.plotly_chart(fig_stress_gen, use_container_width=True)

    # Row 2: Sleep Analysis
    st.markdown("#### 2. Análise de Qualidade do Sono")
    r2_col1, r2_col2 = st.columns(2)
    
    with r2_col1:
        if col_sleep in df_filtered.columns:
            fig_sleep_gen = px.box(
                df_filtered, x='gender', y=col_sleep, color='gender',
                title="Qualidade do Sono por Gênero"
            )
            st.plotly_chart(fig_sleep_gen, use_container_width=True)
            
    with r2_col2:
        if col_sleep in df_filtered.columns:
            fig_sleep_occ = px.box(
                df_filtered, x='occupation', y=col_sleep, color='occupation',
                title="Qualidade do Sono por Ocupação"
            )
            st.plotly_chart(fig_sleep_occ, use_container_width=True)

    # Row 3: Age & Lifestyle
    st.markdown("#### 3. Idade e Estilo de Vida")
    r3_col1, r3_col2 = st.columns(2)
    
    with r3_col1:
        if col_sleep in df_filtered.columns and 'age' in df_filtered.columns:
            fig_sleep_age = px.scatter(
                df_filtered, x='age', y=col_sleep, color=col_stress,
                title="Qualidade do Sono por Idade (Cor=Estresse)",
                trendline="ols"
            )
            st.plotly_chart(fig_sleep_age, use_container_width=True)

    with r3_col2:
        if col_social in df_filtered.columns and col_wellness in df_filtered.columns:
            fig_social = px.scatter(
                df_filtered, x=col_social, y=col_wellness, color=col_stress,
                size=col_sleep,
                title="Bem-Estar vs. Socialização",
                hover_data=['occupation']
            )
            st.plotly_chart(fig_social, use_container_width=True)

    # 4. Mapa de Correlação (Full Width)
    st.markdown("---")
    st.subheader("🔗 Mapa de Correlações Multivariadas")
    plot_correlation_heatmap(df_filtered)

def generate_recommendations(user_input, ideal_profile):
    tips = []
    
    # Screen Time
    if user_input['screen_time'] > ideal_profile['screen_time_hours'] + 1:
        tips.append(f"📉 **Reduza o Tempo de Tela:** Seu tempo ({user_input['screen_time']}h) está acima do ideal ({ideal_profile['screen_time_hours']:.1f}h). Tente pausas a cada hora.")
    
    # Sleep
    if user_input['sleep'] < ideal_profile['sleep_hours'] - 1:
        tips.append(f"😴 **Priorize o Sono:** Você dorme menos ({user_input['sleep']}h) que o grupo de baixo estresse ({ideal_profile['sleep_hours']:.1f}h).")
        
    # Exercise
    if user_input['exercise'] < ideal_profile['exercise_minutes_per_week'] / 60:
        tips.append(f"🏃 **Movimente-se:** O perfil ideal pratica cerca de {ideal_profile['exercise_minutes_per_week']/60:.1f}h de exercícios por semana.")
        
    # Social
    if user_input['social'] < ideal_profile['social_hours_per_week'] - 2:
        tips.append(f"💬 **Socialize:** Interação social ajuda! A média ideal é de {ideal_profile['social_hours_per_week']:.1f}h/semana.")
        
    if not tips:
        tips.append("🌟 **Parabéns!** Seus hábitos estão alinhados com o perfil de alto bem-estar.")
        
    return tips

def render_insights(df):
    st.header("💡 Insights e Relatório Objetivo")
    
    if df.empty:
        st.warning("Sem dados para gerar insights.")
        return

    # Define "Ideal" as Low Stress (< 4) and High Sleep Quality (> 3 if scale 1-5, or just use Sleep Hours > 7)
    # Assuming sleep_quality_1_5 exists, else use sleep_hours
    
    criteria = (df['stress_level_0_10'] <= 4)
    if 'sleep_quality_1_5' in df.columns:
        criteria = criteria & (df['sleep_quality_1_5'] >= 4)
    else:
        criteria = criteria & (df['sleep_hours'] >= 7.5)
        
    df_ideal = df[criteria]
    
    if df_ideal.empty:
        st.info("Não há dados suficientes para definir um perfil 'Ideal' estrito. Mostrando média geral.")
        df_ideal = df
    
    st.markdown("""
    Esta análise compara os hábitos da população geral com o grupo de **"Alto Bem-Estar"** 
    (definido como Estresse ≤ 4/10 e Qualidade do Sono ≥ 4/5).
    """)
    
    # Metrics Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Perfil da População (Média)")
        st.write(f"**Tempo de Tela:** {df['screen_time_hours'].mean():.1f}h")
        st.write(f"**Sono:** {df['sleep_hours'].mean():.1f}h")
        st.write(f"**Exercício:** {df['exercise_minutes_per_week'].mean()/60:.1f}h/sem")
        st.write(f"**Social:** {df['social_hours_per_week'].mean():.1f}h/sem")
        
    with col2:
        st.subheader("🌟 Perfil Ideal (Baixo Estresse)")
        st.write(f"**Tempo de Tela:** {df_ideal['screen_time_hours'].mean():.1f}h")
        st.write(f"**Sono:** {df_ideal['sleep_hours'].mean():.1f}h")
        st.write(f"**Exercício:** {df_ideal['exercise_minutes_per_week'].mean()/60:.1f}h/sem")
        st.write(f"**Social:** {df_ideal['social_hours_per_week'].mean():.1f}h/sem")
    
    st.markdown("---")
    st.subheader("📌 Conclusões Baseadas nos Dados")
    
    diff_screen = df['screen_time_hours'].mean() - df_ideal['screen_time_hours'].mean()
    st.info(f"O grupo de baixo estresse passa em média **{diff_screen:.1f} horas a menos** em frente às telas por dia.")
    
    if 'work_mode' in df.columns:
        best_mode = df_ideal['work_mode'].mode()[0]
        st.success(f"O modo de trabalho mais comum entre pessoas com baixo estresse é: **{best_mode}**.")

def render_playground(df_avg):
    st.header("🧠 Simulador de Bem-Estar (AI)")
    st.markdown("Preveja seu nível de estresse e receba recomendações personalizadas.")

    models = load_models()
    
    # Layout de 2 Colunas
    col_input, col_result = st.columns([1, 1])

    with col_input:
        with st.form("sim_form"):
            st.subheader("Seu Perfil")
            c1, c2 = st.columns(2)
            age = c1.number_input("Idade", 18, 90, 30)
            gender = c2.selectbox("Gênero", ["Male", "Female", "Non-binary/Other"])
            
            c3, c4 = st.columns(2)
            occupation = c3.selectbox("Ocupação", ["Employed", "Self-employed", "Student", "Retired", "Unemployed"])
            work_mode = c4.selectbox("Modelo de Trabalho", ["Remote", "Hybrid", "In-person"])
            
            st.markdown("---")
            st.subheader("Hábitos Diários")
            
            screen_time = st.slider("Tempo Total de Tela (h)", 0.0, 24.0, 8.0)
            work_screen = st.slider("...dos quais trabalho (h)", 0.0, 24.0, 6.0)
            leisure_screen = st.slider("...dos quais lazer (h)", 0.0, 24.0, 2.0)
            
            sleep_hours = st.slider("Horas de Sono", 0.0, 12.0, 7.0)
            productivity = st.slider("Produtividade Percebida (0-10)", 0, 10, 7)
            exercise = st.number_input("Exercício (Horas/Semana)", 0.0, 20.0, 3.0)
            social = st.number_input("Interação Social (Horas/Semana)", 0.0, 50.0, 10.0)
            
            submitted = st.form_submit_button("Rodar Simulação", use_container_width=True)

    if submitted:
        # Feature Engineering
        # FORCE other_screen_hours to 0.0 to avoid exploding gradient due to near-zero variance in scaler
        other_screen = 0.0 
        exercise_mins = exercise * 60
        
        # DataFrame Input
        input_data = pd.DataFrame({
            'age': [age],
            'work_screen_hours': [work_screen],
            'leisure_screen_hours': [leisure_screen],
            'sleep_hours': [sleep_hours],
            'productivity_0_100': [productivity * 10], # Scale 0-10 -> 0-100
            'exercise_minutes_per_week': [exercise_mins],
            'social_hours_per_week': [social],
            'other_screen_hours': [other_screen],
            # One-Hot Encoding
            'gender_Male': [1 if gender == 'Male' else 0],
            'gender_Non-binary/Other': [1 if gender == 'Non-binary/Other' else 0],
            'occupation_Retired': [1 if occupation == 'Retired' else 0],
            'occupation_Self-employed': [1 if occupation == 'Self-employed' else 0],
            'occupation_Student': [1 if occupation == 'Student' else 0],
            'occupation_Unemployed': [1 if occupation == 'Unemployed' else 0],
            'work_mode_In-person': [1 if work_mode == 'In-person' else 0],
            'work_mode_Remote': [1 if work_mode == 'Remote' else 0]
        })

        with col_result:
            st.subheader("Resultados da Análise")
            
            if models:
                # 1. Prediction Block
                try:
                    scaler = models.get('scaler')
                    if scaler:
                        X_scaled = scaler.transform(input_data)
                        stress_pred = models['stress'].predict(X_scaled)[0]
                        sono_pred = models['sono'].predict(X_scaled)[0]
                        
                        # Exibição Visual dos Resultados
                        k1, k2 = st.columns(2)
                        k1.metric("Estresse Previsto", f"{stress_pred:.2f}/10", 
                                  delta="-Bom" if stress_pred < 5 else "+Alto", delta_color="inverse")
                        
                        k2.metric("Qualidade do Sono", f"{sono_pred:.2f}/5", 
                                  delta="+Bom" if sono_pred > 3.5 else "-Baixo")
                        
                        st.progress(min(stress_pred/10, 1.0), text="Nível de Risco de Burnout")
                    else:
                        st.warning("Scaler não encontrado.")
                        stress_pred = 0
                        sono_pred = 0
                except Exception as e:
                    st.error(f"Erro na predição: {e}")
                    st.write("Dados de Entrada:", input_data)
                    stress_pred = 0
                    sono_pred = 0

                # 2. Recommendation Block (Safe from Prediction Errors)
                try:
                    st.markdown("### 📋 Recomendações Personalizadas")
                    
                    # Calculate Ideal Profile from DF for comparison
                    if not df_avg.empty:
                        criteria = (df_avg['stress_level_0_10'] <= 4)
                        if 'sleep_quality_1_5' in df_avg.columns:
                            criteria = criteria & (df_avg['sleep_quality_1_5'] >= 4)
                        
                        # FIX: numeric_only=True prevents string concatenation on object columns
                        ideal_profile = df_avg[criteria].mean(numeric_only=True)
                    else:
                        # Fallback defaults
                        ideal_profile = pd.Series({
                            'screen_time_hours': 5.0,
                            'sleep_hours': 8.0,
                            'exercise_minutes_per_week': 300,
                            'social_hours_per_week': 15
                        })
                        
                    user_metrics = {
                        'screen_time': screen_time,
                        'sleep': sleep_hours,
                        'exercise': exercise,
                        'social': social
                    }
                    
                    tips = generate_recommendations(user_metrics, ideal_profile)
                    for tip in tips:
                        st.info(tip)
                except Exception as e:
                    st.error(f"Erro ao gerar recomendações: {e}")

            else:
                st.info("Modo Demo. Visualizando dados brutos.")

            # Radar Chart
            avg_metrics = {'screen_time': 9.0, 'sleep': 6.5, 'prod': 60, 'social': 8, 'exercise': 2.5}
            if not df_avg.empty:
                 avg_metrics = {
                     'screen_time': df_avg['screen_time_hours'].mean(),
                     'sleep': df_avg['sleep_hours'].mean(),
                     'prod': df_avg['productivity_0_100'].mean(),
                     'social': df_avg['social_hours_per_week'].mean(),
                     'exercise': df_avg['exercise_minutes_per_week'].mean()/60
                 }
            
            user_metrics_radar = {'screen_time': screen_time, 'sleep': sleep_hours, 'prod': productivity, 'social': social, 'exercise': exercise}
            plot_radar_chart(user_metrics_radar, avg_metrics)

# --- MAIN ---

def main():
    df = load_data()
    
    st.sidebar.title("Navegação")
    page = st.sidebar.radio("Ir para:", ["Dashboard Analytics", "AI Simulator", "Insights & Relatório"], index=0)
    
    if page == "Dashboard Analytics":
        render_dashboard(df)
    elif page == "AI Simulator":
        render_playground(df)
    elif page == "Insights & Relatório":
        render_insights(df)

if __name__ == "__main__":
    main()