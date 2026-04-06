# Linear Regression Command Center

A high-fidelity, production-level Streamlit application for deep pedagogical understanding of Linear Regression and Gradient Descent.

## Features
- **Interactive Scatters**: Adjust Slope and Intercept manually.
- **MSE Visualization**: Literal residuals plotted as error bars.
- **3D Loss Landscapes**: Explore the convex optimization basin in real-time.
- **Robust Gradient Descent**: Automated step-by-step optimization with early stopping.
- **Sensitivity Experiments**: Benchmark multiple learning rates ($\alpha$) simultaneously.

## Industrial Design
- **Architecture**: Modular separation of Physics/Logic (`src/core.py`), Visuals (`src/visuals.py`), and Simulation (`src/utils.py`).
- **Robustness**: Custom error handling for divergent models (Infinite/NaN loss).
- **Quality**: Unit-tested core logic using `pytest`.
- **UX**: Premium glassmorphism design with `plotly_dark` themes and Inter typography.

## Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

### 3. Run Tests
```bash
PYTHONPATH=. pytest tests/
```

## Education & Mentorship
Created as a portfolio-level project to demonstrate mastery of:
1. **Vectorized Math**: NumPy-centric implementation.
2. **Convex Optimization**: Understanding the geometry of loss.
3. **Engineering Standards**: Abstract Base Classes, strict typing, and documentation.
