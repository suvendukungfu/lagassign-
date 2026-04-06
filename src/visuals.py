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
        Top-view contour of Loss with optimization path.
        Works best for Degree 1 (m, b space).
        """
        if len(history['w'][0]) > 1:
            # For higher dimensions, we'd need a projection. 
            # For this UI, we'll return an info chart or mock 2D.
            return go.Figure().update_layout(title="Contour only supports Degree 1 (m, b space)")

        # Range around converged params
        final_w = history['w'][-1][0]
        final_b = history['b'][-1]
        
        w_range = np.linspace(final_w - 5, final_w + 5, 50)
        b_range = np.linspace(final_b - 5, final_b + 5, 50)
        W, B = np.meshgrid(w_range, b_range)
        
        Z = np.array([np.mean((xi * X + bi - y)**2) for xi, bi in zip(W.ravel(), B.ravel())]).reshape(W.shape)
        
        fig = go.Figure(data=go.Contour(z=Z, x=w_range, y=b_range, colorscale='Magma'))
        
        # Path Trace
        w_path = [wt[0] for wt in history['w']]
        b_path = history['b']
        fig.add_trace(go.Scatter(x=w_path, y=b_path, mode='lines+markers', name='Path', 
                                marker=dict(size=4, color='white')))
        
        fig.update_layout(**PlottingFactory.THEME, title="Loss Surface Path (Slope vs Intercept)")
        return fig

    @staticmethod
    def dataset_summary(df: pd.DataFrame) -> go.Figure:
        """Pairplot-style preview for real datasets."""
        # Just show histogram of features for now
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['feature'], name='Feature Distribution', marker_color='#00d1ff'))
        fig.update_layout(**PlottingFactory.THEME, title="Internal Data Distribution")
        return fig
