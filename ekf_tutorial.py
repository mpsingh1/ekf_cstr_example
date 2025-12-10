"""
================================================================================
EXTENDED KALMAN FILTER (EKF) FOR CHEMICAL REACTOR ESTIMATION
================================================================================

This tutorial explains how state-space models and Kalman filtering work
using a Continuous Stirred Tank Reactor (CSTR) as an example.

Author: State Space Kinetics Project
================================================================================

TABLE OF CONTENTS
-----------------
1. What is State-Space Modeling?
2. The CSTR System
3. Extended Kalman Filter (EKF) Overview
4. EKF Algorithm Steps
5. Tuning Parameters
6. Key Insights

================================================================================
1. WHAT IS STATE-SPACE MODELING?
================================================================================

State-space models describe dynamic systems using:

    dx/dt = f(x, u)    <- State equation (how states evolve)
    y = h(x)           <- Measurement equation (what we observe)

Where:
    x = state vector (things we want to know but can't directly measure)
    u = input/control (things we can set)
    y = measurements (things we can observe with sensors)

Why use it?
- Unified framework for modeling, control, and estimation
- Handles multiple inputs/outputs naturally
- Foundation for optimal control and filtering

================================================================================
2. THE CSTR SYSTEM
================================================================================

A CSTR is a tank where reactants flow in, react, and products flow out.

States (x):
    - Ca: Concentration of reactant A (mol/L)
    - T:  Reactor temperature (K)

Input (u):
    - Tc: Coolant temperature (K)

Physics:
    
    Material Balance (concentration):
        dCa/dt = (q/V)*(Cai - Ca) - k*Ca
                 ├─────────────┘   └────┘
                 Flow in/out        Reaction consumption
    
    Energy Balance (temperature):
        dT/dt = (q/V)*(Ti - T) + (-dH)*k*Ca/(rho*Cp) + UA*(Tc - T)/(V*rho*Cp)
                ├─────────────┘   ├─────────────────┘   ├─────────────────────┘
                Flow heat         Reaction heat         Cooling/heating
    
    Reaction Rate (Arrhenius):
        k = k0 * exp(-E/(R*T))
        
        Higher temperature → faster reaction → more heat released → even higher temp
        This feedback can cause RUNAWAY if not controlled!

================================================================================
3. EXTENDED KALMAN FILTER (EKF) OVERVIEW
================================================================================

Problem: We want to know Ca (concentration) but we can only measure T (temperature).

Solution: Use the EKF to ESTIMATE Ca from T measurements + physics model.

How it works:

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │   MODEL              MEASUREMENT           ESTIMATE             │
    │   (Physics)    +     (Sensor)        =     (Best guess)         │
    │                                                                 │
    │   "What physics      "What sensor          "Optimal blend       │
    │    predicts"          observes"             of both"            │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

The EKF is "Extended" because it handles NONLINEAR systems by linearizing
around the current estimate at each step.

================================================================================
4. EKF ALGORITHM STEPS
================================================================================

At each time step k:

STEP 1: PREDICT
---------------
    Use the physics model to predict the next state:
    
        x_pred = f(x_hat, u)           # Integrate ODE forward
        P_pred = F @ P @ F.T + Q       # Propagate uncertainty
    
    Where:
        F = Jacobian of f (linearized state transition)
        Q = Process noise covariance (model uncertainty)
        P = State covariance (estimation uncertainty)

STEP 2: UPDATE
--------------
    Correct the prediction using the measurement:
    
        y = sensor reading
        y_pred = h(x_pred)             # What we expect to measure
        residual = y - y_pred          # Innovation (surprise)
        
        S = H @ P_pred @ H.T + R       # Innovation covariance
        K = P_pred @ H.T @ inv(S)      # Kalman gain
        
        x_hat = x_pred + K @ residual  # Corrected estimate
        P = (I - K @ H) @ P_pred       # Updated uncertainty
    
    Where:
        H = Measurement Jacobian (which states we observe)
        R = Measurement noise covariance (sensor uncertainty)
        K = Kalman gain (how much to trust measurement vs model)

================================================================================
5. TUNING PARAMETERS
================================================================================

Three matrices control EKF behavior:

┌─────────────────────────────────────────────────────────────────────────────┐
│  PARAMETER   │  MEANING                    │  EFFECT OF INCREASING         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Q           │  Process noise              │  Trust measurements more,     │
│  (model      │  "How wrong is my model?"   │  faster response,             │
│   uncertainty)                             │  noisier estimates            │
├─────────────────────────────────────────────────────────────────────────────┤
│  R           │  Measurement noise          │  Trust model more,            │
│  (sensor     │  "How noisy are sensors?"   │  slower response,             │
│   uncertainty)                             │  smoother estimates           │
├─────────────────────────────────────────────────────────────────────────────┤
│  P_0         │  Initial covariance         │  Faster initial               │
│  (initial    │  "How uncertain am I at     │  adaptation                   │
│   uncertainty)│   the start?"              │                               │
└─────────────────────────────────────────────────────────────────────────────┘

The Q/R RATIO determines the balance:
    - High Q/R → Fast, responsive, but noisy
    - Low Q/R  → Slow, smooth, but may lag

PRACTICAL TUNING PROCEDURE:
    1. Measure sensor noise to set R (take readings at steady state, compute variance)
    2. Start with small Q
    3. Increase Q until filter tracks disturbances well
    4. If too noisy, increase R or decrease Q
    5. Check that P converges to steady state

================================================================================
6. KEY INSIGHTS
================================================================================

KALMAN GAIN (K):
    - K ≈ 0: Ignore measurement, trust model
    - K ≈ 1: Ignore model, trust measurement
    - K adapts automatically based on uncertainties!

CROSS-COVARIANCE (P_12):
    - Links unobserved states to observed states
    - Negative P_12 means: "If T goes up, Ca probably went down"
    - This is HOW the filter infers Ca from T measurements

THE HIDDEN POWER:
    Even though we only measure temperature, the filter can estimate
    concentration because:
    1. The physics model relates T and Ca (they're coupled)
    2. The Jacobian captures this coupling
    3. The cross-covariance P_12 propagates information

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Simple demonstration of EKF concepts
def demo_kalman_gain_interpretation():
    """
    Demonstrate what Kalman gain means visually.
    
    K blends model prediction and measurement:
        estimate = prediction + K * (measurement - prediction)
        estimate = (1-K)*prediction + K*measurement
    """
    prediction = 10.0
    measurement = 15.0
    
    K_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print("\n" + "="*60)
    print("KALMAN GAIN INTERPRETATION")
    print("="*60)
    print(f"Model prediction: {prediction}")
    print(f"Sensor measurement: {measurement}")
    print("-"*60)
    
    for K in K_values:
        estimate = prediction + K * (measurement - prediction)
        print(f"K = {K:.2f} → Estimate = {estimate:.1f}  ", end="")
        if K == 0:
            print("(100% model)")
        elif K == 1:
            print("(100% sensor)")
        else:
            print(f"({int((1-K)*100)}% model, {int(K*100)}% sensor)")


def demo_qr_ratio_effect():
    """
    Demonstrate how Q/R ratio affects filter behavior.
    """
    print("\n" + "="*60)
    print("Q/R RATIO EFFECT")
    print("="*60)
    
    scenarios = [
        ("High Q, Low R", 1.0, 0.1, "Fast, noisy - trusts sensors"),
        ("Balanced", 0.5, 0.5, "Moderate tracking"),
        ("Low Q, High R", 0.1, 1.0, "Slow, smooth - trusts model"),
    ]
    
    for name, Q, R, description in scenarios:
        ratio = Q / R
        print(f"{name:15} | Q/R = {ratio:5.1f} | {description}")


def run_demo():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("  EKF TUTORIAL - CONCEPTUAL DEMONSTRATIONS")
    print("="*70)
    
    demo_kalman_gain_interpretation()
    demo_qr_ratio_effect()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
    The EKF combines:
    ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
    │  PHYSICS MODEL │  +  │  MEASUREMENTS  │  =  │  BEST ESTIMATE │
    │  (predicts)    │     │  (observes)    │     │  (optimal)     │
    └────────────────┘     └────────────────┘     └────────────────┘
    
    Tuning controls the TRUST balance:
    • Increase Q → trust measurements more
    • Increase R → trust model more
    
    The magic: Even measuring only T, we can estimate Ca!
    """)


if __name__ == "__main__":
    run_demo()

