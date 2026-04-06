import streamlit as st
import ssl
import urllib.request as request
import numpy as np
import pandas as pd
import time

# --- NUCLEAR SSL BYPASS ---
# This force-patches urllib to ignore SSL certificates at the lowest level.
def _patched_urlopen(*args, **kwargs):
    if 'context' not in kwargs:
        kwargs['context'] = ssl._create_unverified_context()
    return request.original_urlopen(*args, **kwargs)

if not hasattr(request, 'original_urlopen'):
    request.original_urlopen = request.urlopen
    request.urlopen = _patched_urlopen
# --------------------------

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
       # --- CACHED HEAVY COMPUTATIONS ---
    @st.cache_data
    def get_surface_data(_eng, _X, _y):
        """Pre-calculates the loss basin once per dataset change."""
        return _eng.compute_loss_surface(_X, _y)

    m_vals_grid, b_vals_grid, Z_grid = get_surface_data(eng, X, y)

    # Tabs Layout
    tab_fit, tab_anim, tab_loss, tab_3d, tab_diag = st.tabs([
        "Performance", 
        "Animation", 
        "Loss Contour", 
        "3D Surface",
        "Explorer"
    ])
    
    with tab_fit:
        col_m1, col_m2 = st.columns([2.5, 1.5])
        
        with col_m2:
            st.markdown("### 📊 Performance Analytics")
            if st.button("Run Full Optimization", use_container_width=True):
                # Using the core engine with Early Stopping
                st.session_state.history = eng.fit_history(X, y, lr=lr, iterations=iters)
                if st.session_state.history and len(st.session_state.history['w']) > 0:
                    st.session_state.learned_w = st.session_state.history['w'][-1]
                    st.session_state.learned_b = st.session_state.history['b'][-1]
            
            if st.session_state.history and st.session_state.learned_w is not None:
                # Calculate metrics for accuracy
                eng.w = st.session_state.learned_w
                eng.b = st.session_state.learned_b
                y_p = eng.predict(X)
                metrics = eng.get_metrics(y, y_p)
                
                # Accuracy Scorecard
                c1, c2 = st.columns(2)
                c1.metric("R² Score", f"{metrics['R2']:.4f}")
                c2.metric("MAE (Error)", f"{metrics['MAE']:.4f}")
                c1.metric("RMSE", f"{metrics['RMSE']:.4f}")
                c2.metric("Final MSE", f"{metrics['MSE']:.4f}")
                
                st.markdown("##### Weights & Coefficients")
                st.dataframe(pd.DataFrame({'Weight': st.session_state.learned_w}), use_container_width=True)
            else:
                st.info("Optimization required")

        with col_m1:
            if st.session_state.history and st.session_state.learned_w is not None:
                # Accuracy Diagnostic Visuals
                sub_tab1, sub_tab2 = st.tabs(["Regression Fit", "Residual (Error) Plot"])
                
                with sub_tab1:
                    fig_fit = PlottingFactory.animated_fit(X, y, {
                        'w': [st.session_state.learned_w], 
                        'b': [st.session_state.learned_b]
                    })
                    st.plotly_chart(fig_fit, use_container_width=True)
                
                with sub_tab2:
                    y_p = eng.predict(X)
                    fig_res = PlottingFactory.residual_plot(y, y_p)
                    st.plotly_chart(fig_res, use_container_width=True)
            else:
                st.info("Run optimization to see deep diagnostics.")

    with tab_anim:
        st.subheader("Real-Time Learning Visualization")
        if st.session_state.history:
            fig_anim = PlottingFactory.animated_fit(X, y, st.session_state.history)
            st.plotly_chart(fig_anim, use_container_width=True)
        else:
            st.warning("Run optimization in the 'Performance' tab first!")

    with tab_loss:
        if poly_deg == 1:
            st.subheader("Top-Down Loss Landscape")
            fig_contour = PlottingFactory.contour_descent(X, y, st.session_state.history)
            st.plotly_chart(fig_contour, use_container_width=True)
        else:
            st.info("Loss contours are limited to 2-parameter models (Degree 1).")

    with tab_3d:
        if poly_deg == 1:
            st.subheader("3D Optimization Basin")
            fig_3d = PlottingFactory.loss_surface_3d(m_vals_grid, b_vals_grid, Z_grid, st.session_state.history)
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("3D surface visualization is limited to 2-parameter models (Degree 1).")

    with tab_diag:
        st.subheader("Training Dataset Audit")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.dataframe(df_raw.head(100), use_container_width=True)
            
            # Export Logic
            if st.session_state.learned_w is not None:
                model_data = {
                    "weights": st.session_state.learned_w.tolist(),
                    "intercept": float(st.session_state.learned_b),
                    "degree": poly_deg,
                    "metrics": metrics if 'metrics' in locals() else "Run optimization first"
                }
                import json
                st.download_button(
                    label="📥 Export Model Weights (JSON)",
                    data=json.dumps(model_data, indent=4),
                    file_name="linear_regression_model.json",
                    mime="application/json",
                    use_container_width=True
                )
        with col_d2:
            fig_dist = PlottingFactory.dataset_summary(df_raw)
            st.plotly_chart(fig_dist, use_container_width=True)

# Footer
st.divider()
st.markdown("Developed by **Antigravity AI (MLE Lead)** | Version **4.0 Production Ready**")
