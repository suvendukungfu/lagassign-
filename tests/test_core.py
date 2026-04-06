import pytest
import numpy as np
from src.core import OptimizationEngine

def test_prediction_identity():
    """Predicting on identity should return weights + bias."""
    # Degree 1: w=[2.0], b=5.0 -> y = 2x + 5
    eng = OptimizationEngine(weights=np.array([2.0]), bias=5.0, degree=1)
    X = np.array([1.0, 2.0, 3.0])
    y_pred = eng.predict(X)
    expected = 2.0 * X + 5.0
    assert np.allclose(y_pred, expected)

def test_loss_calculation():
    """MSE on identical vectors should be zero."""
    eng = OptimizationEngine(degree=1)
    y = np.array([1, 2, 3])
    # Predicted is 0x + 0 = 0. Error is y - 0 = y.
    loss = eng.calculate_loss(y, y)
    assert loss == 0.0

def test_optimization_convergence():
    """Model should converge on a simple line with low noise."""
    X = np.linspace(-5, 5, 100)
    y = 3.0 * X + 2.0
    eng = OptimizationEngine(degree=1)
    # Fit history returns the path
    history = eng.fit_history(X, y, lr=0.01, iterations=1000)
    
    # Check final params from last step in history
    final_w = history['w'][-1]
    final_b = history['b'][-1]
    
    assert abs(final_w[0] - 3.0) < 0.1
    assert abs(final_b - 2.0) < 0.1

def test_divergence_detection():
    """Extremely high learning rate should stop history early."""
    X = np.linspace(-5, 5, 100)
    y = 3.0 * X + 2.0
    eng = OptimizationEngine(degree=1)
    # In V3.0, it breaks the loop instead of raising an error directly during fit_history 
    # to keep the animation smooth. Let's check it stops before max iterations.
    history = eng.fit_history(X, y, lr=100.0, iterations=100)
    assert len(history['loss']) < 100
