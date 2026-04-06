import streamlit as st
import numpy as np
import pandas as pd
import time
import logging

# Refactored Core Imports
from src.core import OptimizationEngine
from src.visuals import PlottingFactory
from src.utils import DataEngine
from src.config import AppConfig

# --- Configuration & UI ---
st.set_page_config(page_title="Deep Linear Regression Command", layout="wide", page_icon="chart")

# Final Style Polish
st.markdown(f"""
<style>
    .stApp {{ background-color: {AppConfig.THEME_COLORS['background']}; color: #fff; }}
    [data-testid="stMetricValue"] {{ font-size: 2.2rem !important; color: {AppConfig.THEME_COLORS['primary']}; }}
    .stMetric {{ background: {AppConfig.THEME_COLORS['card']}; border: 1px solid #333; padding: 1.2rem; border-radius: 12px; }}
    .sidebar .stSlider {{ opacity: 0.9; }}
    .stTabs [data-baseweb="tab"] {{ font-size: 1.1rem; padding: 10px 20px; }}
    .stTabs [aria-selected="true"] {{ background-color: #222 !important; border-bottom: 3px solid {AppConfig.THEME_COLORS['primary']}; }}
</style>
""", unsafe_allow_html=True)

# --- State Management ---
def init_session_state():
    """
    Standardize session state keys to prevent AttributeErrors.
    Initializes keys with safe defaults if they don't exist.
    """
    defaults = {
        'history': None,
        'learned_w': None,
        'learned_b': 0.0,
        'data_ver': 0,
        'last_config': ""
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Sidebar: Control Center ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/512/statistics.png", width=80)
    st.header("Laboratory Control")
    
    # 1. Dataset Selection
    ds_mode = st.selectbox("Dataset Selection", 
                        ["Simulated Linear", "California Housing (MedInc)", "Diabetes (BMI)", "CSV Upload"])
    
    df_raw = pd.DataFrame()
    if ds_mode == "Simulated Linear":
        n_s = st.slider("Samples", 50, 1000, 200)
        noise = st.slider("Noise", 0.0, 10.0, 2.0)
        df_raw = DataEngine.generate_simulated(n=n_s, noise=noise)
    elif ds_mode == "California Housing (MedInc)":
        df_raw = DataEngine.load_preset("california")
    elif ds_mode == "Diabetes (BMI)":
        df_raw = DataEngine.load_preset("diabetes")
    elif ds_mode == "CSV Upload":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            df_csv = pd.read_csv(uploaded_file)
            cols = list(df_csv.columns)
            x_sel = st.selectbox("Select X (Feature)", cols)
            y_sel = st.selectbox("Select y (Target)", cols)
            # Safe selection
            df_raw = pd.DataFrame({'feature': df_csv[x_sel], 'target': df_csv[y_sel]})

    st.divider()
    
    # 2. Model Selection
    st.subheader("Model Architecture")
    poly_deg = st.number_input("Polynomial Degree", 1, 10, 1)
    reg_mode = st.radio("Regularization", ["None", "Ridge (L2)", "Lasso (L1)"], horizontal=True)
    reg_alpha = st.slider("Regularization Strength (λ)", 0.0, 5.0, 0.1)
    
    # 3. Solver Params
    st.subheader("SGD Parameters")
    lr = st.number_input("Learning Rate (α)", 0.0001, 1.0, 0.05, format="%.4f")
    iters = st.number_input("Epochs", 10, 1000, 100)

    # --- Change Detection Logic ---
    # Moved below inputs to ensure variables are defined
    current_config = f"{ds_mode}-{poly_deg}-{reg_mode}-{reg_alpha}"
    if st.session_state.last_config != current_config:
        st.session_state.history = None
        st.session_state.learned_w = None
        st.session_state.last_config = current_config

# --- Data Engine Logic ---
if not df_raw.empty:
    X, y = DataEngine.preprocess(df_raw, 'feature', 'target', scale=True)
    
    # Hero Title
    st.title("Advanced Regression Command Center")
    st.markdown(f"Mode: **{ds_mode}** | Architecture: **Degree {poly_deg} {reg_mode}**")
    
    # Tabs Layout
    tab_fit, tab_anim, tab_loss, tab_diag = st.tabs([
        "Model Fit & Diagnostic", 
        "Gradient Descent Animation", 
        "Loss Geometry", 
        "Dataset Exploration"
    ])

    # Pre-calculate Optimization for All Tabs
    eng = OptimizationEngine(degree=poly_deg, alpha=reg_alpha, mode=reg_mode.replace(" (L1)", "").replace(" (L2)", ""))
    
    with tab_fit:
        col_m1, col_m2 = st.columns([3, 1])
        
        with col_m2:
            st.markdown("### 📊 Metrics")
            if st.button("Run Full Optimization", use_container_width=True):
                st.session_state.history = eng.fit_history(X, y, lr=lr, iterations=iters)
                if st.session_state.history and len(st.session_state.history['w']) > 0:
                    st.session_state.learned_w = st.session_state.history['w'][-1]
                    st.session_state.learned_b = st.session_state.history['b'][-1]
            
            if st.session_state.history and st.session_state.learned_w is not None:
                final_loss = st.session_state.history['loss'][-1]
                st.metric("Final Loss (MSE)", f"{final_loss:.4f}")
                st.markdown("##### Learned Weights")
                st.dataframe(pd.DataFrame({'Weight': st.session_state.learned_w}), 
                           use_container_width=True)
            else:
                st.info("Optimization required")

        with col_m1:
            if st.session_state.history and st.session_state.learned_w is not None:
                # Update engine with learned params for prediction
                eng.w = st.session_state.learned_w
                eng.b = st.session_state.learned_b
                y_p = eng.predict(X) 
                
                fig_fit = PlottingFactory.animated_fit(X, y, {
                    'w': [st.session_state.learned_w], 
                    'b': [st.session_state.learned_b]
                })
                st.plotly_chart(fig_fit, use_container_width=True)
            else:
                st.info("Adjust hyperparameters and click 'Run Optimization' to start.")

    with tab_anim:
        st.subheader("Real-Time Learning Visualization")
        if st.session_state.history:
            fig_anim = PlottingFactory.animated_fit(X, y, st.session_state.history)
            st.plotly_chart(fig_anim, use_container_width=True)
            st.markdown("""
            **What to Observe:**
            - **Momentum**: Watch how the line 'swings' towards the data.
            - **Bias Shift**: The line often finds the correct intercept ($b$) before perfectly aligning its slope ($m$).
            - **Polynomial Curves**: For Degree > 1, watch how the line 'bends' to capture curvature.
            """)
        else:
            st.warning("Run optimization in the 'Model Fit' tab first!")

    with tab_loss:
        if poly_deg == 1:
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                st.subheader("3D Gradient Surface")
                # Showing static surface for performance, with path overlay if desired
                # Placeholder for 3D trajectory
                st.info("Visualizing the convex basin in parameter space.")
            with col_l2:
                st.subheader("Contour Level Curves")
                if st.session_state.history:
                    fig_contour = PlottingFactory.contour_descent(X, y, st.session_state.history)
                    st.plotly_chart(fig_contour, use_container_width=True)
        else:
            st.info("3D surface visualization is limited to 2-parameter models (Degree 1).")
            st.markdown(f"**Current Model Dimensions:** {poly_deg} weights + 1 bias = {poly_deg+1}D space.")

    with tab_diag:
        st.subheader("Training Dataset Audit")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.dataframe(df_raw.head(100), use_container_width=True)
        with col_d2:
            fig_dist = PlottingFactory.dataset_summary(df_raw)
            st.plotly_chart(fig_dist, use_container_width=True)

# Footer
st.divider()
st.markdown("Developed by **Antigravity AI (MLE Lead)** | Version **3.0 Stable**")
