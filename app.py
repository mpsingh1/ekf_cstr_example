import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

# ==========================================
# 1. STREAMLIT UI & PARAMETERS
# ==========================================
st.set_page_config(page_title="EKF CSTR Estimator", layout="wide")

st.title("Extended Kalman Filter for Chemical Reactor")
st.markdown("""
**The Challenge:** Estimate reactor **Concentration** when we can only measure **Temperature**.
The EKF combines a physics model with noisy sensor data to produce optimal state estimates.
""")

# ==========================================
# REFERENCE SECTIONS (COLLAPSIBLE)
# ==========================================

# ----- CSTR SYSTEM -----
with st.expander("CSTR System Diagram & Equations", expanded=False):
    # CSTR Diagram
    fig_cstr = go.Figure()
    
    # Reactor vessel
    fig_cstr.add_shape(type="rect", x0=2, y0=1, x1=6, y1=5,
                       line=dict(color="black", width=3), fillcolor="lightblue")
    
    # Inlet pipe
    fig_cstr.add_shape(type="line", x0=0, y0=4.5, x1=2, y1=4.5, line=dict(color="black", width=2))
    fig_cstr.add_annotation(x=1, y=4.8, text="Feed In", showarrow=False, font=dict(size=11))
    fig_cstr.add_annotation(x=1, y=4.2, text="q, CAi, Ti", showarrow=False, font=dict(size=9, color="blue"))
    
    # Outlet pipe
    fig_cstr.add_shape(type="line", x0=6, y0=1.5, x1=8, y1=1.5, line=dict(color="black", width=2))
    fig_cstr.add_annotation(x=7, y=1.8, text="Product Out", showarrow=False, font=dict(size=11))
    fig_cstr.add_annotation(x=7, y=1.2, text="q, CA, T", showarrow=False, font=dict(size=9, color="red"))
    
    # Cooling jacket
    fig_cstr.add_shape(type="rect", x0=1.5, y0=0.5, x1=6.5, y1=5.5,
                       line=dict(color="green", width=2, dash="dash"), fillcolor="rgba(0,255,0,0.1)")
    fig_cstr.add_shape(type="line", x0=1.5, y0=0.8, x1=0.5, y1=0.8, line=dict(color="green", width=2))
    fig_cstr.add_annotation(x=0.8, y=0.5, text="Coolant Tc", showarrow=False, font=dict(size=9, color="green"))
    
    # Stirrer
    fig_cstr.add_shape(type="line", x0=4, y0=5.5, x1=4, y1=3, line=dict(color="gray", width=3))
    fig_cstr.add_shape(type="line", x0=3.3, y0=2.5, x1=4.7, y1=2.5, line=dict(color="gray", width=3))
    
    # Labels
    fig_cstr.add_annotation(x=4, y=3.5, text="V, CA, T", showarrow=False, font=dict(size=12, color="darkblue"))
    fig_cstr.add_annotation(x=4, y=1.8, text="A → B", showarrow=False, font=dict(size=11, color="red"))
    
    fig_cstr.update_layout(showlegend=False, height=350, margin=dict(l=10,r=10,t=10,b=10),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 8.5]),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 6.5]),
                           plot_bgcolor='white')
    st.plotly_chart(fig_cstr, use_container_width=True)
    
    # Balance equations
    col_mat, col_eng = st.columns(2)
    with col_mat:
        st.markdown("**Material Balance:**")
        st.latex(r"\frac{dC_A}{dt} = \frac{q}{V}(C_{Ai} - C_A) - kC_A")
        st.markdown("**State Vector:**")
        st.latex(r"\mathbf{x} = \begin{bmatrix} C_A \\ T \end{bmatrix}")
    with col_eng:
        st.markdown("**Energy Balance:**")
        st.latex(r"\frac{dT}{dt} = \frac{q}{V}(T_i - T) + \frac{(-\Delta H)}{\rho C_p}kC_A + \frac{UA}{V\rho C_p}(T_c - T)")
        st.markdown("**Arrhenius Rate:**")
        st.latex(r"k = k_0 \exp\left(-\frac{E}{RT}\right)")

# ----- EKF MATHEMATICS -----
with st.expander("EKF Mathematics", expanded=False):
    col_pred, col_upd = st.columns(2)
    
    with col_pred:
        st.markdown("### Predict Step")
        st.markdown("**State Prediction:**")
        st.latex(r"\hat{\mathbf{x}}_{k|k-1} = f(\hat{\mathbf{x}}_{k-1}, \mathbf{u}_k)")
        st.markdown("**Covariance Prediction:**")
        st.latex(r"\mathbf{P}_{k|k-1} = \mathbf{F}_k \mathbf{P}_{k-1} \mathbf{F}_k^T + \mathbf{Q}")
        st.markdown("**Jacobian (Linearization):**")
        st.latex(r"\mathbf{F} = \mathbf{I} + \mathbf{A} \Delta t")
    
    with col_upd:
        st.markdown("### Update Step")
        st.markdown("**Innovation:**")
        st.latex(r"\mathbf{r}_k = \mathbf{y}_k - \mathbf{H}\hat{\mathbf{x}}_{k|k-1}")
        st.markdown("**Kalman Gain:**")
        st.latex(r"\mathbf{K} = \mathbf{P}_{k|k-1}\mathbf{H}^T(\mathbf{H}\mathbf{P}_{k|k-1}\mathbf{H}^T + \mathbf{R})^{-1}")
        st.markdown("**State Update:**")
        st.latex(r"\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}\mathbf{r}_k")
        st.markdown("**Covariance Update:**")
        st.latex(r"\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}_{k|k-1}")
    
    st.divider()
    st.markdown("### Jacobian Matrix")
    st.latex(r"\mathbf{A} = \begin{bmatrix} -\frac{q}{V} - k & -C_A k \frac{E}{RT^2} \\ \frac{(-\Delta H)}{\rho C_p} k & -\frac{q}{V} + \frac{(-\Delta H)}{\rho C_p} C_A \frac{\partial k}{\partial T} - \frac{UA}{V\rho C_p} \end{bmatrix}")
    
    st.markdown("### Kalman Gain Interpretation")
    col_k0, col_k1 = st.columns(2)
    with col_k0:
        st.latex(r"\mathbf{K} \to 0")
        st.caption("Trust MODEL (ignore sensor)")
    with col_k1:
        st.latex(r"\mathbf{K} \to 1")
        st.caption("Trust SENSOR (ignore model)")

# ----- GLOSSARY -----
with st.expander("Glossary of Terms", expanded=False):
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("#### State Variables")
        st.markdown("""
| Symbol | Name | Units | Description |
|--------|------|-------|-------------|
| $C_A$ | Concentration | mol/L | Reactant concentration |
| $T$ | Temperature | K | Reactor temperature |
| $\\mathbf{x}$ | State vector | - | $[C_A, T]^T$ |
""")
        
        st.markdown("#### Reactor Parameters")
        st.markdown("""
| Symbol | Name | Units | Description |
|--------|------|-------|-------------|
| $q$ | Flow rate | L/min | Volumetric flow |
| $V$ | Volume | L | Reactor volume |
| $C_{Ai}$ | Inlet conc | mol/L | Feed concentration |
| $T_i$ | Inlet temp | K | Feed temperature |
| $T_c$ | Coolant temp | K | Jacket temperature |
| $UA$ | Heat transfer | W/K | Coefficient × area |
""")
        
        st.markdown("#### Reaction Kinetics")
        st.markdown("""
| Symbol | Name | Units | Description |
|--------|------|-------|-------------|
| $k$ | Rate constant | 1/min | Arrhenius rate |
| $k_0$ | Pre-exponential | 1/min | Frequency factor |
| $E$ | Activation energy | J/mol | Energy barrier |
| $\\Delta H$ | Enthalpy | J/mol | Reaction heat |
""")
    
    with col_g2:
        st.markdown("#### EKF Variables")
        st.markdown("""
| Symbol | Name | Description |
|--------|------|-------------|
| $\\hat{\\mathbf{x}}$ | State estimate | Best guess of $[C_A, T]^T$ |
| $\\mathbf{P}$ | Covariance | Uncertainty in estimate |
| $\\mathbf{K}$ | Kalman gain | Correction weight |
| $\\mathbf{Q}$ | Process noise | Model uncertainty |
| $\\mathbf{R}$ | Measurement noise | Sensor uncertainty |
| $\\mathbf{F}$ | State transition | Linearized dynamics |
| $\\mathbf{H}$ | Measurement matrix | Observation model |
| $P_{12}$ | Cross-covariance | Ca-T correlation |
""")
        
        st.markdown("#### Tuning Effects")
        st.markdown("""
| Increase | Effect |
|----------|--------|
| $Q$ | Faster, noisier (trust sensor) |
| $R$ | Slower, smoother (trust model) |
| $Q/R$ ratio | Higher = more responsive |
""")
        
        st.markdown("#### Acronyms")
        st.markdown("""
| Acronym | Full Name |
|---------|-----------|
| CSTR | Continuous Stirred Tank Reactor |
| EKF | Extended Kalman Filter |
| RMSE | Root Mean Square Error |
""")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Settings")
sim_time = st.sidebar.slider("Simulation Time (s)", 5.0, 60.0, 10.0, step=1.0)
dt = 0.1
steps = int(sim_time / dt)

st.sidebar.header("Reactor Parameters")
col_a, col_b = st.sidebar.columns(2)
with col_a:
    q = st.number_input("Flow rate q (L/min)", 1.0, 500.0, 10.0, step=10.0)
    V = st.number_input("Volume V (L)", 1.0, 500.0, 10.0, step=10.0)
    Cai = st.number_input("Inlet Conc Cai (mol/L)", 0.1, 10.0, 1.0, step=0.1)
with col_b:
    Ti = st.number_input("Inlet Temp Ti (K)", 300.0, 400.0, 350.0, step=5.0)
    UA = st.number_input("Heat Transfer UA (W/K)", 1000.0, 200000.0, 50000.0, step=5000.0)

st.sidebar.header("Reaction Kinetics")
col_c, col_d = st.sidebar.columns(2)
with col_c:
    dH = st.number_input("Enthalpy ΔH (J/mol)", -500000.0, 0.0, -50000.0, step=10000.0)
    E = st.number_input("Activation E (J/mol)", 10000.0, 100000.0, 50000.0, step=5000.0)
with col_d:
    k0_exp = st.number_input("k₀ exponent (10^x)", 3.0, 12.0, 6.86, step=0.5)
    k0 = 10**k0_exp

st.sidebar.header("Coolant Profile")
tc_initial = st.sidebar.number_input("Initial Coolant Temp (K)", 280.0, 400.0, 300.0)
tc_final = st.sidebar.number_input("Final Coolant Temp (K)", 280.0, 450.0, 350.0)
tc_step_time = st.sidebar.slider("Step Change Time (s)", 1.0, sim_time-1, 5.0)

st.sidebar.header("Initial Conditions")
Ca_init = st.sidebar.number_input("Initial Ca (mol/L)", 0.0, 2.0, 0.5, step=0.1)
T_init = st.sidebar.number_input("Initial T (K)", 300.0, 450.0, 350.0, step=5.0)

st.sidebar.header("EKF Tuning")
st.sidebar.markdown("**Process Noise Q** (model uncertainty)")
q_ca = st.sidebar.slider("Q_Ca (concentration)", -8.0, -1.0, -4.0, format="1e%.0f")
q_t = st.sidebar.slider("Q_T (temperature)", -4.0, 2.0, 0.0, format="1e%.0f")

st.sidebar.markdown("**Measurement Noise R** (sensor uncertainty)")
r_ca = st.sidebar.slider("R_Ca", -6.0, 0.0, -3.7, format="1e%.1f")
r_t = st.sidebar.slider("R_T", -2.0, 2.0, 0.3, format="1e%.1f")

st.sidebar.header("Measurement Mode")
meas_mode = st.sidebar.radio("What can we measure?", ["Temperature Only", "Both Ca and T"])

# Fixed constants
rho = 1000.0    # Density (g/L)
Cp = 4.184      # Heat capacity (J/(g·K))
R_gas = 8.314   # Gas constant (J/(mol·K))

def cstr_dynamics(t, x, u_Tc):
    """The Nonlinear Physics Model"""
    Ca, T = x
    Ca = max(0.0, Ca)
    T = max(273.0, T)
    
    # Arrhenius rate law
    k = k0 * np.exp(-E / (R_gas * T))
    
    # Molar reaction rate: r = k * Ca [mol/(L·min)]
    r = k * Ca
    
    # Material Balance: dCa/dt = inflow - outflow - consumption
    dCadt = (q/V) * (Cai - Ca) - r
    
    # Energy Balance: dT/dt = flow + reaction heat + cooling
    # Reaction heat = r × (-ΔH) / (ρ·Cp) [K/min]
    dTdt = (q/V) * (Ti - T) + r * (-dH) / (rho * Cp) + (UA / (V * rho * Cp)) * (u_Tc - T)
    
    return [dCadt, dTdt]

def get_jacobian(x, u_Tc, dt):
    """Linearize the model around state x for the EKF."""
    Ca, T = x
    Ca = max(0.0, Ca)
    T = max(273.0, T)
    
    k = k0 * np.exp(-E / (R_gas * T))
    dk_dT = k * (E / (R_gas * T**2))
    
    # Coefficients
    alpha = (-dH) / (rho * Cp)  # Reaction heat coefficient
    beta = UA / (V * rho * Cp)   # Cooling coefficient
    
    # Continuous Jacobian A = ∂f/∂x
    # f1 = dCa/dt, f2 = dT/dt
    a11 = -q/V - k              # ∂f1/∂Ca
    a12 = -Ca * dk_dT           # ∂f1/∂T
    a21 = alpha * k             # ∂f2/∂Ca
    a22 = -q/V + alpha * Ca * dk_dT - beta  # ∂f2/∂T
    
    A = np.array([[a11, a12], [a21, a22]])
    
    # Discretize: F = I + A*dt
    F = np.eye(2) + A * dt
    return F

# Build matrices from sliders
Q_mat = np.diag([10**q_ca, 10**q_t])
R_Ca_val = 10**r_ca
R_T_val = 10**r_t

# ==========================================
# 3. SIMULATION LOOP
# ==========================================
t_span = np.linspace(0, sim_time, steps)

# Initial Conditions (from sidebar)
x_true = np.array([Ca_init, T_init])   # True state
x_model = np.array([Ca_init, T_init])  # Pure model (no noise)
x_est = np.array([Ca_init, T_init])    # EKF estimate
P = np.diag([0.1, 100.0])              # Initial covariance

# Storage
history = {
    't': [], 'Tc': [],
    'Ca_true': [], 'T_true': [],
    'Ca_model': [], 'T_model': [],
    'Ca_est': [], 'T_est': [],
    'K_Ca': [], 'P_Ca': [], 'P_T': [], 'P_12': [],
    'k_rate': [],       # Reaction rate
    'Q_flow': [],       # Heat from flow (K/min)
    'Q_rxn': [],        # Heat from reaction (K/min)
    'Q_cool': []        # Heat from cooling (K/min)
}

# Measurement setup
if meas_mode == "Both Ca and T":
    H = np.eye(2)
    R_mat = np.diag([R_Ca_val, R_T_val])
else:
    H = np.array([[0, 1]])
    R_mat = np.array([[R_T_val]])

# Run Simulation
for k in range(steps):
    t = k * dt
    
    # Coolant temperature (step change)
    Tc = tc_initial if t < tc_step_time else tc_final
    
    # A. PURE MODEL (no noise reference)
    sol_model = solve_ivp(lambda t, x: cstr_dynamics(t, x, Tc), [0, dt], x_model, 
                          method='RK45', rtol=1e-6, atol=1e-8)
    x_model = sol_model.y[:, -1]
    x_model[0] = max(0.0, x_model[0])
    x_model[1] = max(273.0, x_model[1])
    
    # B. TRUE SYSTEM (with process noise)
    sol_true = solve_ivp(lambda t, x: cstr_dynamics(t, x, Tc), [0, dt], x_true,
                         method='RK45', rtol=1e-6, atol=1e-8)
    x_true = sol_true.y[:, -1]
    x_true = x_true + np.random.multivariate_normal([0, 0], Q_mat)
    x_true[0] = max(0.0, x_true[0])
    x_true[1] = max(273.0, x_true[1])
    
    # C. GENERATE MEASUREMENTS
    if meas_mode == "Both Ca and T":
        y_meas = np.array([
            x_true[0] + np.random.normal(0, np.sqrt(R_Ca_val)),
            x_true[1] + np.random.normal(0, np.sqrt(R_T_val))
        ])
    else:
        y_meas = np.array([x_true[1] + np.random.normal(0, np.sqrt(R_T_val))])
    
    # D. EKF PREDICT
    sol_est = solve_ivp(lambda t, x: cstr_dynamics(t, x, Tc), [0, dt], x_est,
                        method='RK45', rtol=1e-6, atol=1e-8)
    x_pred = sol_est.y[:, -1]
    x_pred[0] = max(0.0, x_pred[0])
    x_pred[1] = max(273.0, x_pred[1])
    
    F = get_jacobian(x_est, Tc, dt)
    P_pred = F @ P @ F.T + Q_mat
    
    # E. EKF UPDATE
    y_pred = H @ x_pred
    residual = y_meas - y_pred
    
    S = H @ P_pred @ H.T + R_mat
    K = P_pred @ H.T @ np.linalg.inv(S)
    
    x_est = x_pred + K @ residual
    x_est[0] = max(0.0, x_est[0])
    x_est[1] = max(273.0, x_est[1])
    P = (np.eye(2) - K @ H) @ P_pred
    
    # Calculate reaction rate (Arrhenius)
    T_curr = max(273.0, x_true[1])
    Ca_curr = max(0.0, x_true[0])
    k_rate = k0 * np.exp(-E / (R_gas * T_curr))
    
    # Molar conversion rate: r = k * Ca [mol/(L·min)]
    r_molar = k_rate * Ca_curr
    
    # Calculate ALL heat terms (K/min)
    Q_flow = (q / V) * (Ti - T_curr)                 # Heat from flow [K/min]
    Q_rxn = r_molar * (-dH) / (rho * Cp)            # Heat from reaction [K/min]
    Q_cool = (UA / (V * rho * Cp)) * (Tc - T_curr)  # Heat from cooling [K/min]
    
    # Store Data
    history['t'].append(t)
    history['Tc'].append(Tc)
    history['Ca_true'].append(x_true[0])
    history['T_true'].append(x_true[1])
    history['Ca_model'].append(x_model[0])
    history['T_model'].append(x_model[1])
    history['Ca_est'].append(x_est[0])
    history['T_est'].append(x_est[1])
    history['K_Ca'].append(K[0, 0])
    history['P_Ca'].append(P[0, 0])
    history['P_T'].append(P[1, 1])
    history['P_12'].append(P[0, 1])
    history['k_rate'].append(k_rate)
    history['Q_flow'].append(Q_flow)
    history['Q_rxn'].append(Q_rxn)
    history['Q_cool'].append(Q_cool)

df = pd.DataFrame(history)

# ==========================================
# 4. VISUALIZATION
# ==========================================
tab1, tab2, tab3 = st.tabs(["State Estimates", "EKF Internals", "Reaction Dynamics"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Concentration Estimate")
        fig_ca = go.Figure()
        
        fig_ca.add_trace(go.Scatter(x=df['t'], y=df['Ca_true'], mode='lines',
                                    name='True (noisy)', line=dict(color='black', width=2)))
        fig_ca.add_trace(go.Scatter(x=df['t'], y=df['Ca_model'], mode='lines',
                                    name='Pure Model', line=dict(color='blue', width=2)))
        fig_ca.add_trace(go.Scatter(x=df['t'], y=df['Ca_est'], mode='lines',
                                    name='EKF Estimate', line=dict(color='red', width=3, dash='dash')))
        
        # Confidence interval
        sigma = np.sqrt(df['P_Ca'])
        fig_ca.add_trace(go.Scatter(x=df['t'], y=df['Ca_est'] + 2*sigma, mode='lines',
                                    line=dict(width=0), showlegend=False))
        fig_ca.add_trace(go.Scatter(x=df['t'], y=df['Ca_est'] - 2*sigma, mode='lines',
                                    fill='tonexty', fillcolor='rgba(255,0,0,0.15)', 
                                    line=dict(width=0), name='95% Confidence'))
        
        fig_ca.add_vline(x=tc_step_time, line_dash="dot", line_color="gray", annotation_text="Coolant Step")
        fig_ca.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0),
                             xaxis_title="Time (s)", yaxis_title="Concentration (mol/L)",
                             legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        st.plotly_chart(fig_ca, use_container_width=True)
    
    with col2:
        st.subheader("Temperature Estimate")
        fig_t = go.Figure()
        
        fig_t.add_trace(go.Scatter(x=df['t'], y=df['T_true'], mode='lines',
                                   name='True (noisy)', line=dict(color='black', width=2)))
        fig_t.add_trace(go.Scatter(x=df['t'], y=df['T_model'], mode='lines',
                                   name='Pure Model', line=dict(color='blue', width=2)))
        fig_t.add_trace(go.Scatter(x=df['t'], y=df['T_est'], mode='lines',
                                   name='EKF Estimate', line=dict(color='red', width=3, dash='dash')))
        fig_t.add_trace(go.Scatter(x=df['t'], y=df['Tc'], mode='lines',
                                   name='Coolant', line=dict(color='green', width=1.5, dash='dot')))
        
        fig_t.add_vline(x=tc_step_time, line_dash="dot", line_color="gray")
        fig_t.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0),
                            xaxis_title="Time (s)", yaxis_title="Temperature (K)",
                            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        st.plotly_chart(fig_t, use_container_width=True)

with tab2:
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Kalman Gain (K_Ca)")
        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(x=df['t'], y=df['K_Ca'], mode='lines',
                                   line=dict(color='green', width=2)))
        fig_k.add_vline(x=tc_step_time, line_dash="dot", line_color="gray")
        fig_k.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0),
                            xaxis_title="Time (s)", yaxis_title="K_Ca")
        st.plotly_chart(fig_k, use_container_width=True)
        st.caption("How much the filter trusts measurements for concentration updates. K→0: trust model, K→1: trust sensor.")
    
    with col4:
        st.subheader("Cross-Covariance P12")
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=df['t'], y=df['P_12'], mode='lines',
                                   line=dict(color='purple', width=2)))
        fig_p.add_vline(x=tc_step_time, line_dash="dot", line_color="gray")
        fig_p.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0),
                            xaxis_title="Time (s)", yaxis_title="P₁₂")
        st.plotly_chart(fig_p, use_container_width=True)
        st.caption("Coupling between Ca and T estimates. Negative = inverse relationship (higher T → lower Ca).")

with tab3:
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("Reaction Rate (Arrhenius)")
        fig_rate = go.Figure()
        fig_rate.add_trace(go.Scatter(x=df['t'], y=df['k_rate'], mode='lines',
                                      name='k (1/min)', line=dict(color='orange', width=2)))
        fig_rate.add_vline(x=tc_step_time, line_dash="dot", line_color="gray")
        fig_rate.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0),
                               xaxis_title="Time (s)", yaxis_title="k (1/min)")
        st.plotly_chart(fig_rate, use_container_width=True)
        st.caption("k = k₀ × exp(-E/RT). Higher temp → exponentially faster reaction.")
    
    with col6:
        st.subheader("Heat Balance (dT/dt = Flow + Rxn + Cool)")
        fig_heat = go.Figure()
        fig_heat.add_trace(go.Scatter(x=df['t'], y=df['Q_flow'], mode='lines',
                                      name='Flow Heat', line=dict(color='green', width=2)))
        fig_heat.add_trace(go.Scatter(x=df['t'], y=df['Q_rxn'], mode='lines',
                                      name='Reaction Heat', line=dict(color='red', width=2)))
        fig_heat.add_trace(go.Scatter(x=df['t'], y=df['Q_cool'], mode='lines',
                                      name='Cooling Heat', line=dict(color='blue', width=2)))
        # True net heat = dT/dt (should be ~0 at steady state)
        net_heat = np.array(df['Q_flow']) + np.array(df['Q_rxn']) + np.array(df['Q_cool'])
        fig_heat.add_trace(go.Scatter(x=df['t'], y=net_heat, mode='lines',
                                      name='Net (dT/dt)', line=dict(color='black', width=3, dash='dash')))
        fig_heat.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_heat.add_vline(x=tc_step_time, line_dash="dot", line_color="gray")
        fig_heat.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0),
                               xaxis_title="Time (s)", yaxis_title="Heat Rate (K/min)",
                               legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("Green=flow, Red=reaction, Blue=cooling. Black dashed=NET (≈0 at steady state).")

# ==========================================
# 5. METRICS & INSIGHTS
# ==========================================
st.divider()

col_m1, col_m2, col_m3 = st.columns(3)
rmse_ca = np.sqrt(np.mean((df['Ca_true'] - df['Ca_est'])**2))
rmse_t = np.sqrt(np.mean((df['T_true'] - df['T_est'])**2))
final_k = df['K_Ca'].iloc[-1]

col_m1.metric("Ca RMSE", f"{rmse_ca:.4f} mol/L")
col_m2.metric("T RMSE", f"{rmse_t:.2f} K")
col_m3.metric("Steady-State K_Ca", f"{final_k:.4f}")

st.markdown("""
### Key Insights

| What You See | What It Means |
|--------------|---------------|
| **Black line** | True system (with process noise) |
| **Blue line** | Pure model (no noise, reference) |
| **Red dashed** | EKF estimate (optimal blend) |
| **Red shaded** | 95% confidence interval |

**Tuning Guide:**
- **Increase Q** → Filter trusts measurements more (faster, noisier)
- **Increase R** → Filter trusts model more (slower, smoother)
- **K_Ca → 0** → Ignoring measurements, trusting model
- **K_Ca → 1** → Ignoring model, trusting measurements
""")
