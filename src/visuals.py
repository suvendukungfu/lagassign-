import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Union

class PlottingFactory:
    """
    Advanced Visualization Factory for Regression animations and higher-level diagnostics.
    """
    
    THEME: Dict[str, Any] = {
        'template': 'plotly_dark',
        'margin': dict(l=20, r=20, t=50, b=20),
        'font': dict(family="Inter, sans-serif", size=13),
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)'
    }

    @staticmethod
    def animated_fit(X: np.ndarray, y: np.ndarray, history: Dict[str, List]) -> go.Figure:
        """
        Creates a Plotly Animation showing the fit line changing over iterations.
        """
        # Downsample for smooth web display
        step = max(1, len(history['w']) // 50)
        w_hist = history['w'][::step]
        b_hist = history['b'][::step]
        
        fig = go.Figure()
        
        # 1. Base Scatter (Static)
        fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Actual', 
                                marker=dict(color='#00d1ff', size=10, opacity=0.4)))
        
        # 2. Prediction Line (Dynamic)
        # Using polynomial expansion for fit line prediction
        def get_pred(w, b, deg):
            X_p = np.column_stack([X**i for i in range(1, deg + 1)])
            return X_p @ w + b

        if not w_hist:
            return go.Figure().update_layout(title="No optimization history to display")
            
        deg = len(w_hist[0])
        y_initial = get_pred(w_hist[0], b_hist[0], deg)
        fig.add_trace(go.Scatter(x=X, y=y_initial, mode='lines', name='Model Evolution',
                                line=dict(color='#ff4b4b', width=4)))
        
        # 3. Create Animation Frames
        frames = []
        for i in range(len(w_hist)):
            y_p = get_pred(w_hist[i], b_hist[i], deg)
            frames.append(go.Frame(data=[
                go.Scatter(x=X, y=y),               # Marker Trace
                go.Scatter(x=X, y=y_p)             # Line Trace
            ], name=str(i)))
            
        fig.update_layout(
            **PlottingFactory.THEME,
            title="Gradient Descent: Model Evolution",
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50}}]),
                         dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}}])]
            )]
        )
        fig.frames = frames
        return fig

    @staticmethod
    def contour_descent(X: np.ndarray, y: np.ndarray, history: Dict[str, List]) -> go.Figure:
        """
        Animated top-view contour of Loss with moving optimization point.
        """
        if not history or len(history['w']) == 0 or len(history['w'][0]) > 1:
            return go.Figure().update_layout(**PlottingFactory.THEME, title="Contour supports Degree 1 only")

        # Range around converged params
        final_w, final_b = history['w'][-1][0], history['b'][-1]
        w_range = np.linspace(final_w - 5, final_w + 5, 50)
        b_range = np.linspace(final_b - 5, final_b + 5, 50)
        W, B = np.meshgrid(w_range, b_range)
        Z = np.array([np.mean(((xi * X + bi) - y)**2) for xi, bi in zip(W.ravel(), B.ravel())]).reshape(W.shape)

        fig = go.Figure(data=go.Contour(z=Z, x=w_range, y=b_range, colorscale='Magma', name='Loss Surface'))
        
        m_path = [wt[0] for wt in history['w']]
        b_path = history['b']
        skip = max(1, len(m_path) // 40)
        m_anim, b_anim = m_path[::skip], b_path[::skip]

        # Path trace (static background)
        fig.add_trace(go.Scatter(x=m_anim, y=b_anim, mode='lines', name='Path', 
                                line=dict(color='white', dash='dot', width=1)))
        # Moving Point
        fig.add_trace(go.Scatter(x=[m_anim[0]], y=[b_anim[0]], mode='markers', name='State',
                                marker=dict(size=12, color='#00d1ff', symbol='diamond')))

        frames = [go.Frame(data=[go.Contour(z=Z, x=w_range, y=b_range),
                                go.Scatter(x=m_anim[:i+1], y=b_anim[:i+1]),
                                go.Scatter(x=[m_anim[i]], y=[b_anim[i]])],
                          name=str(i)) for i in range(len(m_anim))]
        
        fig.update_layout(**PlottingFactory.THEME, title="Loss Landscape Animation",
            updatemenus=[dict(type="buttons", buttons=[
                dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50}}]),
                dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}}])
            ])])
        fig.frames = frames
        return fig

    @staticmethod
    def dataset_summary(df: pd.DataFrame) -> go.Figure:
        """Pairplot-style preview for real datasets."""
        # Just show histogram of features for now
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['feature'], name='Feature Distribution', marker_color='#00d1ff'))
        fig.update_layout(**PlottingFactory.THEME, title="Internal Data Distribution")
        return fig

    @staticmethod
    def residual_plot(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        """Visualizes the errors: Predicted vs Residual."""
        residuals = y_true - y_pred
        fig = go.Figure()
        
        # Zero Line
        fig.add_shape(type="line", x0=y_pred.min(), y0=0, x1=y_pred.max(), y1=0,
                      line=dict(color="white", width=2, dash="dash"))
        
        # Scatter
        fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers',
                                 marker=dict(color='#ff4b4b', size=8, opacity=0.6),
                                 name='Residuals'))
                                 
        fig.update_layout(**PlottingFactory.THEME, title="Residual Analysis (Errors vs Predicted)",
                          xaxis_title="Predicted Value", yaxis_title="Residual (Error)")
        return fig

    @staticmethod
    def loss_surface_3d(m_vals: np.ndarray, b_vals: np.ndarray, Z: np.ndarray, 
                        history: Optional[Dict[str, List]] = None) -> go.Figure:
        """
        Animated 3D surface plot with a moving 'Descent' point.
        """
        fig = go.Figure(data=[go.Surface(z=Z, x=m_vals, y=b_vals, colorscale='Viridis', opacity=0.8, showscale=False)])
        
        if history and len(history['w']) > 0 and len(history['w'][0]) == 1:
            m_path, b_path, z_path = [wt[0] for wt in history['w']], history['b'], history['loss']
            skip = max(1, len(m_path) // 40)
            m_a, b_a, z_a = m_path[::skip], b_path[::skip], z_path[::skip]

            # Static Path Trace
            fig.add_trace(go.Scatter3d(x=m_a, y=b_a, z=z_a, mode='lines', 
                                     line=dict(color='white', width=4), name='Descent Path'))
            # Animated Explorer
            fig.add_trace(go.Scatter3d(x=[m_a[0]], y=[b_a[0]], z=[z_a[0]], mode='markers', 
                                     marker=dict(size=8, color='#ff4b4b'), name='Explorer'))

            frames = [go.Frame(data=[go.Surface(z=Z, x=m_vals, y=b_vals),
                                    go.Scatter3d(x=m_a[:i+1], y=b_a[:i+1], z=z_a[:i+1]),
                                    go.Scatter3d(x=[m_a[i]], y=[b_a[i]], z=[z_a[i]])],
                              name=str(i)) for i in range(len(m_a))]
            fig.frames = frames

        fig.update_layout(**PlottingFactory.THEME, title="3D Optimization Terrain",
            scene=dict(xaxis_title="Slope (m)", yaxis_title="Intercept (b)", zaxis_title="Loss"),
            updatemenus=[dict(type="buttons", buttons=[
                dict(label="Animate Descent", method="animate", args=[None, {"frame": {"duration": 50}}])
            ])])
        return fig

    @staticmethod
    def learning_curve(losses: List[float]) -> go.Figure:
        """Visualizes how the model improved over time."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=losses, mode='lines', line=dict(color='#00d1ff', width=3)))
        fig.update_layout(**PlottingFactory.THEME, title="Learning Curve (Loss vs Iterations)",
                          xaxis_title="Iteration", yaxis_title="Loss (MSE)")
        return fig
