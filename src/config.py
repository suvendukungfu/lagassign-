"""
Configuration and Constants for the Linear Regression Command Center.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    # Page Settings
    TITLE: str = "Linear Regression Command Center"
    ICON: str = "chart"
    PORT: int = 8502
    
    # Data Simulation Defaults
    DEFAULT_SAMPLES: int = 100
    DEFAULT_NOISE: float = 2.0
    DEFAULT_OUTLIERS: int = 2
    
    # ML Defaults
    DEFAULT_LEARNING_RATE: float = 0.01
    DEFAULT_ITERATIONS: int = 100
    CONVERGENCE_THRESHOLD: float = 1e-6
    DIVERGENCE_THRESHOLD: float = 1e15
    
    # Visual Styles
    THEME_COLORS = {
        'primary': '#00d1ff',
        'secondary': '#ff4b4b',
        'background': '#0e1117',
        'card': '#1a1c24',
        'text': '#ffffff'
    }
