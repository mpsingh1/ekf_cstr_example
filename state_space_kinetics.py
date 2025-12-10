import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

# ==========================================
# 1. SETUP & PHYSICS
# ==========================================
# Physical Constants (Realistic CSTR parameters)
q = 100.0      # Flow rate (L/min)
V = 100.0      # Reactor volume (L)
rho = 1000.0   # Density (g/L)
Cp = 4.184     # Heat capacity (J/(g·K)) - water-like
dH = -500000.0  # Reaction enthalpy (J/mol) - exothermic
E = 50000.0    # Activation energy (J/mol)
R = 8.314      # Gas constant (J/(mol·K))
k0 = 7.2e6     # Pre-exponential factor (1/min) - more realistic
UA = 500000.0  # Heat transfer coefficient × area (W/K) - HIGH to make coolant dominant
Ti = 350.0     # Inlet temperature (K) - match initial T
Cai = 1.0      # Inlet concentration (mol/L)

# Simulation mode: 'model_only' or 'ekf'
SIM_MODE = 'ekf'  # Change to 'ekf' for full EKF simulation

def cstr_dynamics(t, x, u):
    Ca, T = x
    # Physical constraints
    Ca = max(0.0, Ca)
    T = max(273.0, T)  # Minimum 0°C
    
    # Arrhenius reaction rate
    k = k0 * np.exp(-E / (R * T))
    
    # Material balance: dCa/dt
    dCa = (q/V) * (Cai - Ca) - k * Ca
    
    # Energy balance: dT/dt
    # Convert dH from J/mol to J/L by assuming molecular weight ~100 g/mol
    MW = 100.0  # Approximate molecular weight (g/mol)
    dT = (q/V) * (Ti - T) + (-dH / (MW * rho * Cp)) * k * Ca + (UA / (V * rho * Cp)) * (u - T)
    
    return [dCa, dT]

def get_continuous_jacobian(x, u):
    Ca, T = x
    Ca = max(0.0, Ca)
    T = max(273.0, T)
    
    k = k0 * np.exp(-E / (R * T))
    T_term = k * (E / (R * T**2))
    
    MW = 100.0
    alpha = (-dH / (MW * rho * Cp))
    beta = (UA / (V * rho * Cp))
    
    # Jacobian Matrix A
    # Row 1: dCa/dCa, dCa/dT
    # Row 2: dT/dCa,  dT/dT
    A = np.array([[-q/V - k,       -Ca * T_term], 
                  [alpha * k,      -q/V + alpha * Ca * T_term - beta]])
    return A

# ==========================================
# 2. SIMULATION SETUP
# ==========================================
dt = 0.1
steps = 100
time = np.linspace(0, steps*dt, steps)

# State Initialization (realistic starting point)
x_real = np.array([0.8, 350.0])  # Initial: Ca=0.8 mol/L, T=350K
x_hat  = np.array([0.8, 350.0])  # EKF estimate - START WITH ERROR to see convergence
P      = np.diag([0.1, 100.0])   # Initial Uncertainty - higher = more adaptive

# Tuning - CRITICAL FOR PERFORMANCE
Q = np.diag([1e-4, 1])   # Process Noise: very small for clean true signal
R_Ca = 2e-4                # Concentration measurement noise variance
R_T  = 2                 # Temperature measurement noise variance

# Measurement mode: 'T_only' or 'both'
MEAS_MODE = 'both'  # 'T_only' = only temperature sensor, 'both' = Ca and T sensors

# Storage for Visualization
history = {
    'Ca_real': [], 'Ca_est': [], 'Ca_model': [],  # model = pure ODE, no noise
    'T_real': [],  'T_est': [],  'T_model': [],
    'K_Ca': [],    # Kalman gain for concentration
    'P_link': [],
    'Tc': []       # Coolant temperature
}

# ==========================================
# 3. SIMULATION LOOP
# ==========================================
if SIM_MODE == 'model_only':
    print("Simulating model dynamics only...")
    
    for k in tqdm(range(steps), desc="Model Simulation", unit="step"):
        # Coolant temperature profile: step change at t=5s
        Tc = 300.0 if k < 50 else 350.0
        
        # Integrate ODE
        sol = solve_ivp(cstr_dynamics, [0, dt], x_real, args=(Tc,), 
                        method='RK45', rtol=1e-6, atol=1e-8,
                        dense_output=False)
        x_real = sol.y[:, -1]
        
        # Physical constraints
        x_real[0] = max(0.0, x_real[0])
        x_real[1] = max(273.0, x_real[1])
        
        # Logging
        history['Ca_real'].append(x_real[0])
        history['T_real'].append(x_real[1])
        history['Tc'].append(Tc)
        
else:  # EKF mode
    print(f"Simulating with EKF (measurement mode: {MEAS_MODE})...")
    
    # Pure model state (no noise, for reference)
    x_model = np.array([0.8, 350.0])
    
    for k in tqdm(range(steps), desc="EKF Simulation", unit="step"):
        # Coolant temperature profile: step change at t=5s
        Tc = 300.0 if k < 50 else 350.0
        
        # 0. PURE MODEL (no noise, reference)
        sol_model = solve_ivp(cstr_dynamics, [0, dt], x_model, args=(Tc,), 
                              method='RK45', rtol=1e-6, atol=1e-8,
                              dense_output=False)
        x_model = sol_model.y[:, -1]
        x_model[0] = max(0.0, x_model[0])
        x_model[1] = max(273.0, x_model[1])
        
        # 1. REAL SYSTEM (Truth)
        sol = solve_ivp(cstr_dynamics, [0, dt], x_real, args=(Tc,), 
                        method='RK45', rtol=1e-6, atol=1e-8,
                        dense_output=False)
        x_real = sol.y[:, -1]
        x_real[0] = max(0.0, x_real[0])
        x_real[1] = max(273.0, x_real[1])
        
        # Add process noise
        x_real = x_real + np.random.multivariate_normal([0,0], Q)
        x_real[0] = max(0.0, x_real[0])
        x_real[1] = max(273.0, x_real[1])
        
        # Generate measurements
        if MEAS_MODE == 'both':
            y_meas = np.array([
                x_real[0] + np.random.normal(0, np.sqrt(R_Ca)),  # Ca measurement
                x_real[1] + np.random.normal(0, np.sqrt(R_T))    # T measurement
            ])
            H = np.eye(2)  # Measure both states
            R_mat = np.diag([R_Ca, R_T])
        else:  # T_only
            y_meas = np.array([x_real[1] + np.random.normal(0, np.sqrt(R_T))])
            H = np.array([[0, 1]])  # Only measure temperature
            R_mat = np.array([[R_T]])
        
        # 2. EKF PREDICT
        sol_est = solve_ivp(cstr_dynamics, [0, dt], x_hat, args=(Tc,),
                            method='RK45', rtol=1e-6, atol=1e-8,
                            dense_output=False)
        x_pred = sol_est.y[:, -1]
        x_pred[0] = max(0.0, x_pred[0])
        x_pred[1] = max(273.0, x_pred[1])
        
        A = get_continuous_jacobian(x_hat, Tc)
        F = np.eye(2) + A * dt
        P_pred = F @ P @ F.T + Q
        
        # 3. EKF UPDATE
        y_pred = H @ x_pred
        residual = y_meas - y_pred
        
        S = H @ P_pred @ H.T + R_mat
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        x_hat = x_pred + K @ residual
        x_hat[0] = max(0.0, x_hat[0])
        x_hat[1] = max(273.0, x_hat[1])
        P = (np.eye(2) - K @ H) @ P_pred
        
        # 4. LOGGING
        history['Ca_real'].append(x_real[0])
        history['Ca_est'].append(x_hat[0])
        history['Ca_model'].append(x_model[0])
        history['T_real'].append(x_real[1])
        history['T_est'].append(x_hat[1])
        history['T_model'].append(x_model[1])
        history['K_Ca'].append(K[0, 0])
        history['P_link'].append(P[0,1])
        history['Tc'].append(Tc)

# ==========================================
# 4. VISUALIZATION DASHBOARD
# ==========================================
if SIM_MODE == 'model_only':
    # Simple model dynamics plot
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Plot 1: Concentration
    axs[0].set_title("Concentration Dynamics", fontweight='bold')
    axs[0].plot(time, history['Ca_real'], 'b-', linewidth=2, label='Concentration')
    axs[0].axvline(5.0, color='r', linestyle='--', alpha=0.7, label='Cooling Failure')
    axs[0].set_ylabel('Ca (mol/L)')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: Temperature
    axs[1].set_title("Temperature Dynamics", fontweight='bold')
    axs[1].plot(time, history['T_real'], 'r-', linewidth=2, label='Reactor Temperature')
    axs[1].plot(time, history['Tc'], 'g--', linewidth=1.5, alpha=0.7, label='Coolant Temperature')
    axs[1].axvline(5.0, color='r', linestyle='--', alpha=0.7)
    axs[1].set_ylabel('Temperature (K)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # Plot 3: Reaction Rate
    k_vals = [k0 * np.exp(-E / (R * T)) for T in history['T_real']]
    axs[2].set_title("Reaction Rate", fontweight='bold')
    axs[2].plot(time, k_vals, 'm-', linewidth=2)
    axs[2].axvline(5.0, color='r', linestyle='--', alpha=0.7)
    axs[2].set_ylabel('k (1/min)')
    axs[2].set_xlabel('Time (s)')
    axs[2].grid(True, alpha=0.3)
    
else:
    # Full EKF dashboard
    fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    
    # Plot 1: Concentration Estimate
    axs[0].set_title("Concentration Estimate", fontweight='bold')
    axs[0].plot(time, history['Ca_real'], 'k-', label='True (noisy)', linewidth=2, alpha=0.7)
    axs[0].plot(time, history['Ca_model'], 'b-', linewidth=2, label='Pure Model')
    axs[0].plot(time, history['Ca_est'], 'r--', linewidth=2, label='EKF Estimate')
    axs[0].axvline(5.0, color='gray', linestyle=':', alpha=0.5, label='Coolant Step')
    axs[0].set_ylabel('Ca (mol/L)')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: Temperature Estimate
    axs[1].set_title("Temperature Estimate", fontweight='bold')
    axs[1].plot(time, history['T_real'], 'k-', label='True (noisy)', linewidth=2, alpha=0.7)
    axs[1].plot(time, history['T_model'], 'b-', linewidth=2, label='Pure Model')
    axs[1].plot(time, history['T_est'], 'r--', linewidth=2, label='EKF Estimate')
    axs[1].plot(time, history['Tc'], 'g:', linewidth=1.5, alpha=0.7, label='Coolant')
    axs[1].axvline(5.0, color='gray', linestyle=':', alpha=0.5)
    axs[1].set_ylabel('T (K)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, alpha=0.3)
    
    # Plot 3: Kalman Gain for Concentration
    axs[2].set_title("Kalman Gain for Concentration (K_Ca)", fontweight='bold')
    axs[2].plot(time, history['K_Ca'], 'g-', linewidth=2)
    axs[2].axvline(5.0, color='gray', linestyle=':', alpha=0.5)
    axs[2].set_ylabel('K_Ca')
    axs[2].grid(True, alpha=0.3)
    
    # Plot 4: Covariance
    axs[3].set_title("Cross-Covariance P_12 (Ca-T coupling)", fontweight='bold')
    axs[3].plot(time, history['P_link'], 'm-', linewidth=2)
    axs[3].axvline(5.0, color='gray', linestyle=':', alpha=0.5)
    axs[3].set_ylabel('P_12')
    axs[3].set_xlabel('Time (s)')
    axs[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show(block=False)
plt.savefig('state_space_kinetics.png')
print(f"Plot saved as 'state_space_kinetics.png'")