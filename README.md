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

## Files

- `app.py` - Main Streamlit application with interactive simulator
- `state_space_kinetics.py` - Standalone EKF simulation script with matplotlib
- `ekf_tutorial.py` - Tutorial with conceptual demonstrations
- `pages/1_EKF_Math.py` - Detailed mathematical explanations
- `pages/2_Glossary.py` - Complete glossary of terms
- `pages/3_CSTR_Diagram.py` - System diagram and balance equations

## Installation

```bash
pip install streamlit numpy pandas plotly scipy tqdm matplotlib
```

## Usage

### Interactive App
```bash
streamlit run app.py
```

### Standalone Simulation
```bash
python state_space_kinetics.py
```

### Tutorial
```bash
python ekf_tutorial.py
```

## System Equations

**Material Balance:**
```
dCa/dt = (q/V)(Cai - Ca) - kCa
```

**Energy Balance:**
```
dT/dt = (q/V)(Ti - T) + (-ΔH/ρCp)kCa + (UA/VρCp)(Tc - T)
```

**Arrhenius Rate:**
```
k = k0 * exp(-E/RT)
```

## EKF Algorithm

1. **Predict**: Propagate state and uncertainty using physics model
2. **Update**: Correct prediction using sensor measurements
3. **Iterate**: Repeat for optimal state estimation

## Tuning Parameters

- **Q (Process Noise)**: Model uncertainty - higher values trust measurements more
- **R (Measurement Noise)**: Sensor uncertainty - higher values trust model more
- **P0 (Initial Covariance)**: Initial uncertainty - affects convergence speed

## Key Insights

- The EKF can estimate unmeasured states (concentration) from measured states (temperature)
- Cross-covariance P12 captures the coupling between concentration and temperature
- Kalman gain adaptively balances trust between model predictions and sensor measurements

## License

MIT

## Author
Manjot Pal Singh
Created for educational purposes to demonstrate Extended Kalman Filtering.

