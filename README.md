# Extended Kalman Filter for Chemical Reactor

An interactive simulation and visualization tool for understanding Extended Kalman Filter (EKF) applied to a Continuous Stirred Tank Reactor (CSTR).

## Overview

This project demonstrates state estimation using an Extended Kalman Filter on a nonlinear chemical reactor system. The challenge: estimate reactor concentration when we can only measure temperature.

## Features

- **Interactive Streamlit App**: Adjust reactor parameters, kinetics, and EKF tuning in real-time
- **CSTR System Diagram**: Visual representation of the reactor with all components
- **EKF Mathematics**: Complete mathematical derivations and explanations
- **Comprehensive Glossary**: All terms, symbols, and parameters explained
- **Multiple Visualization Modes**:
  - State estimates (concentration and temperature)
  - EKF internals (Kalman gain, covariance)
  - Reaction dynamics (reaction rate, heat balance)

## Project Structure

```
state_space/
├── app.py                    # Main Streamlit application
├── models/                   # Dynamical system models
│   ├── __init__.py
│   └── cstr.py              # CSTR reactor model
├── filters/                  # State estimation algorithms
│   ├── __init__.py
│   └── ekf.py               # Extended Kalman Filter
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── coolant_profiles.py  # Disturbance generators
│   └── plotting.py          # Chart styling
├── pages/                    # Streamlit pages
│   ├── 1_EKF_Math.py
│   ├── 2_Glossary.py
│   └── 3_CSTR_Diagram.py
├── requirements.txt         # Python dependencies
└── README.md
```

## Architecture

The project is designed for scalability and extensibility:

### Models (`models/`)
Add new dynamical systems (e.g., pendulum, vehicle) by implementing:
- `dynamics(t, x, u)` - State derivatives
- `jacobian(x, u, dt)` - Linearization for EKF
- `get_state_names()` - For plotting
- `get_state_units()` - For axis labels

### Filters (`filters/`)
The EKF is model-agnostic and can work with any model following the interface above. You can add other estimators like UKF or Particle Filters.

### Utils (`utils/`)
Reusable components:
- **Coolant Profiles**: Various disturbance scenarios (step, sinusoidal, random walk, etc.)
- **Plotting**: Economist magazine-inspired chart styling for clean, professional visuals

## Live Demo

https://ekfcstrexample-vczdzrrivhe8wpucqmlnfx.streamlit.app/

## Installation

```bash
pip install -r requirements.txt
```

## Running Locally

```bash
streamlit run app.py
```

## Author

Manjot Singh  
Email: smanjotpal@gmail.com
