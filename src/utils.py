import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union
import logging
from sklearn.datasets import fetch_california_housing, load_diabetes

logger = logging.getLogger(__name__)

class DataEngine:
    """
    Advanced Data Engine for simulation, real dataset loading, and preprocessing.
    """
    
    @staticmethod
    def generate_simulated(n: int = 100, m: float = 2.5, b: float = 5.0, 
                          noise: float = 2.0, seed: int = 42) -> pd.DataFrame:
        """Generates synthetic linear data as a DataFrame."""
        np.random.seed(seed)
        X = np.linspace(-10, 10, n)
        y = m * X + b + np.random.normal(0, noise, n)
        return pd.DataFrame({'feature': X, 'target': y})

    @staticmethod
    def load_preset(name: str = "california") -> pd.DataFrame:
        """Loads a real-world dataset from sklearn."""
        if name == "california":
            data = fetch_california_housing()
            # Feature: MedInc (Median Income), Target: MedHouseVal
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            return df[['MedInc', 'target']].rename(columns={'MedInc': 'feature'})
        elif name == "diabetes":
            data = load_diabetes()
            # BMI vs Disease Progression
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            return df[['bmi', 'target']].rename(columns={'bmi': 'feature'})
        return pd.DataFrame()

    @staticmethod
    def preprocess(df: pd.DataFrame, x_col: str, y_col: str, 
                   scale: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Cleans, handles missing values, and scales data."""
        df = df.dropna(subset=[x_col, y_col])
        X = df[x_col].values
        y = df[y_col].values
        
        if scale:
            # Standard Scaling (Z-score)
            X = (X - np.mean(X)) / (np.std(X) + 1e-10)
            
        return X.astype(np.float64), y.astype(np.float64)
