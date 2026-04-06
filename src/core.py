import numpy as np
from typing import Tuple, List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class OptimizationEngine:
    """
    Highly optimized multi-dimensional Gradient Descent.
    Supports Linear, Polynomial, and Regularized models through weight vectorization.
    """
    
    def __init__(self, weights: Optional[np.ndarray] = None, bias: float = 0.0, 
                 degree: int = 1, alpha: float = 0.0, mode: str = 'none'):
        self.w = weights if weights is not None else np.zeros(degree)
        self.b = float(bias)
        self.degree = int(degree)
        self.reg_alpha = float(alpha)
        self.mode = mode.lower() # 'none', 'ridge', 'lasso'

    def _expand_features(self, X: np.ndarray) -> np.ndarray:
        """Transforms scalar X into polynomial features vector [x, x^2, ..., x^deg]."""
        X = np.array(X).flatten()
        return np.column_stack([X**i for i in range(1, self.degree + 1)])

    def _validate_inputs(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Ensures consistent array types and shapes."""
        X_arr = np.array(X).astype(np.float64)
        y_arr = np.array(y).astype(np.float64).flatten() if y is not None else None
        return X_arr, y_arr

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        """Prediction: y = sum(w_i * x^i) + b."""
        X, _ = self._validate_inputs(X_raw)
        X_poly = self._expand_features(X)
        return X_poly @ self.w + self.b

    def calculate_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """MSE + Regularization Penalty."""
        mse = float(np.mean((y_true - y_pred)**2))
        
        # Penalties
        if self.mode == 'ridge':
            penalty = self.reg_alpha * np.sum(self.w**2)
        elif self.mode == 'lasso':
            penalty = self.reg_alpha * np.sum(np.abs(self.w))
        else:
            penalty = 0.0
            
        return mse + penalty

    def calculate_gradients(self, X_poly: np.ndarray, y: np.ndarray, 
                            w: np.ndarray, b: float) -> Tuple[np.ndarray, float]:
        """Vectorized analytic gradients for weights and bias."""
        n = len(y)
        y_p = X_poly @ w + b
        error = y_p - y
        
        # dw = (2/n) * X^T (y_p - y)
        dw = (2.0 / n) * (X_poly.T @ error)
        # db = (2/n) * sum(y_p - y)
        db = (2.0 / n) * np.sum(error)
        
        # Add Regularization Gradients
        if self.mode == 'ridge':
            dw += 2.0 * self.reg_alpha * w
        elif self.mode == 'lasso':
            dw += self.reg_alpha * np.sign(w)
            
        return dw, db

    def gradient_descent_step(self, X_poly: np.ndarray, y: np.ndarray, 
                              w: np.ndarray, b: float, lr: float) -> Tuple[np.ndarray, float]:
        """Perform a single iteration of optimization."""
        dw, db = self.calculate_gradients(X_poly, y, w, b)
        new_w = w - lr * dw
        new_b = b - lr * db
        return new_w, new_b

    def fit_history(self, X_raw: np.ndarray, y_raw: np.ndarray, 
                    lr: float, iterations: int) -> Dict[str, List]:
        """
        Calculates optimization trajectory with Early Stopping and Smart Initialization.
        """
        X, y = self._validate_inputs(X_raw, y_raw)
        X_poly = self._expand_features(X)
        
        # 1. Smart Initialization (Small random weights)
        np.random.seed(42)
        w = np.random.randn(self.degree) * 0.01
        b = 0.0
        
        history: Dict[str, List] = {'w': [], 'b': [], 'loss': []}
        best_loss = float('inf')
        patience = 8
        tol = 1e-7
        
        for i in range(iterations):
            y_p = X_poly @ w + b
            loss = self.calculate_loss(y, y_p)
            
            # 2. Explosive Divergence Check
            if np.isnan(loss) or np.isinf(loss) or loss > 1e18:
                break
                
            history['w'].append(w.copy())
            history['b'].append(b)
            history['loss'].append(loss)
            
            # 3. Early Stopping Logic
            if abs(best_loss - loss) < tol:
                patience -= 1
                if patience == 0: break
            else:
                patience = 8
            
            if loss < best_loss:
                best_loss = loss

            # 4. Perform Step
            w, b = self.gradient_descent_step(X_poly, y, w, b, lr)
            
        return history
