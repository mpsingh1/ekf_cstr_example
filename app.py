import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

# Import modular components
from models import CSTRModel
from filters import ExtendedKalmanFilter
from utils import generate_coolant_profile, economist_chart_style, get_colors

# Get theme-aware colors (refresh on each run)
COLORS = get_colors()

# ==========================================
# 1. STREAMLIT UI & PARAMETERS
# ==========================================
st.set_page_config(
    page_title="The Estimation Challenge | Extended Kalman Filter",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme-aware CSS - Use Streamlit's actual theme colors
try:
    theme_base = st.get_option("theme.base")
    bg_color = st.get_option("theme.backgroundColor")
    secondary_bg = st.get_option("theme.secondaryBackgroundColor")
    text_color_theme = st.get_option("theme.textColor")
    is_dark = theme_base == "dark"
except:
    is_dark = False
    bg_color = "#ffffff"
    secondary_bg = "#f0f2f6"
    text_color_theme = "#1a1a1a"

if is_dark:
    text_color = text_color_theme
    subtitle_color = "#b0b0b0"
    h3_color = "#e0e0e0"
    sidebar_bg = secondary_bg  # Use Streamlit's actual sidebar color
    info_bg = "#1a2a35"
    info_border = "#0077b6"
    warning_bg = "#3a2e1a"
    warning_border = "#ff4444"
    success_bg = "#1a2e1f"
    success_border = "#4caf50"
    estimate_color = "#ff4444"
else:
    text_color = text_color_theme
    subtitle_color = "#666"
    h3_color = "#2a2a2a"
    sidebar_bg = secondary_bg  # Use Streamlit's actual sidebar color
    info_bg = "#e8f4f8"
    info_border = "#0077b6"
    warning_bg = "#fff8e6"
    warning_border = "#E3120B"
    success_bg = "#e8f5e9"
    success_border = "#2e7d32"
    estimate_color = "#E3120B"

st.markdown(f"""
<style>
    /* Typography - Economist uses Econ Sans */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {{
        font-family: 'Inter', sans-serif;
    }}
    
    /* Main title styling */
    h1 {{
        font-weight: 700;
        color: {text_color};
        font-size: 2.5rem !important;
        line-height: 1.2;
        margin-bottom: 0.5rem !important;
    }}
    
    /* Subtitle styling */
    .subtitle {{
        font-size: 1.1rem;
        color: {subtitle_color};
        font-weight: 400;
        line-height: 1.6;
        margin-bottom: 2rem;
    }}
    
    /* Section headers */
    h2 {{
        font-weight: 600;
        color: {estimate_color};
        font-size: 1.5rem !important;
        margin-top: 2rem !important;
        border-bottom: 3px solid {estimate_color};
        padding-bottom: 0.5rem;
    }}
    
    h3 {{
        font-weight: 600;
        color: {h3_color};
        font-size: 1.2rem !important;
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg};
    }}
    
    /* Info boxes */
    .stInfo {{
        background-color: {info_bg};
        border-left: 4px solid {info_border};
    }}
    
    /* Warning boxes */
    .stWarning {{
        background-color: {warning_bg};
        border-left: 4px solid {warning_border};
    }}
    
    /* Success boxes */
    .stSuccess {{
        background-color: {success_bg};
        border-left: 4px solid {success_border};
    }}
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {{
        font-size: 1.8rem;
        font-weight: 600;
        color: {estimate_color};
    }}
    
    /* Remove extra padding */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1>The Estimation Challenge | Extended Kalman Filter</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">How do you measure what you cannot see? This interactive demonstration shows how the Extended Kalman Filter infers unmeasured concentration from temperature readings in a chemical reactor—combining physics, probability, and real-time data.</p>', unsafe_allow_html=True)

# Key insight box - Economist style
st.markdown("""
<div style="background-color: #fff8e6; padding: 1.5rem; border-left: 4px solid #E3120B; margin: 1.5rem 0;">
    <h3 style="margin-top: 0; color: #E3120B; font-size: 1.1rem;">The simulation paradox</h3>
    <p style="margin-bottom: 0; color: #333;">In this demonstration, we show the "true state"—what is actually happening in the reactor. <strong>In reality, you would never have this.</strong> When measuring only temperature, you must infer concentration through physics and statistics. The grey line is our answer key; the orange dots and red estimate are what you would actually have. This is why state estimation matters.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("💡 Tips for Better EKF Tracking"):
    st.markdown("""
    If the red line (EKF) isn't tracking the gray line (truth) well:
    
    **NEW:** True system noise and EKF tuning are now **decoupled**!
    - "True System Noise" section: Controls physical reality (⬛ gray line)
    - "EKF Tuning" section: Controls filter behavior (🟥 red line)
    
    ### Quick Fixes:
    
    1. **Use "Good Tracking" preset** (in EKF Tuning)
       - Pre-configured Q_ekf and R_ekf for good performance
       
    2. **Match EKF to Reality** (best practice)
       - Set EKF Q_Ca = True Q_Ca
       - Set EKF R_T = True R_T
       - This is "perfect tuning" (rarely possible in real life!)
       
    3. **Increase EKF Q_Ca** manually (-4 → -2)
       - Makes EKF less confident in model → adapts faster
       
    4. **Decrease EKF R_T** (0.3 → 0)
       - Makes EKF trust sensor more → follows measurements closer
       
    5. **Start with wrong initial estimate** 
       - Set "Est. Ca" to 0.3 when true is 0.5
       - Watch convergence!
       
    6. **Test Mismatch Scenarios** (realistic)
       - Set True Q_Ca = -3, EKF Q_Ca = -5 (overconfident filter)
       - Watch poor tracking!
       
    7. **Use Random Seed** (for fair comparison)
       - Keep seed constant (e.g., 42)
       - Change only EKF parameters
       - True state stays identical → see pure effect of tuning!
       
    8. **Test with Coolant Disturbances** (realistic scenarios)
       - Try "Sinusoidal" coolant → periodic disturbance
       - Try "Random Walk" → stochastic but smooth changes  
       - Watch how EKF handles real operational disturbances!
    
    **Key insight:** In reality, you never know Q_true or R_true. You must 
    estimate them through system identification or empirical tuning. Coolant 
    variations are often the dominant disturbance in real processes!
    """)

with st.expander("🔬 Understanding TRUE vs EKF Noise Parameters"):
    st.markdown("""
    ### The Key Distinction: Reality vs Belief
    
    This app **decouples** physical reality from EKF tuning:
    
    ```
    PHYSICAL REALITY                    EKF's WORLD
    ================                    ============
    
    True Reactor                        EKF's Model
         ↓                                   ↓
    + Q_true (actual disturbances)     Q_ekf (assumed Q)
         ↓                                   ↓
    = True State (⬛ black/gray)        Predicted State
         ↓                                   ↓
    Sensor                              
         ↓                                   ↓
    + R_true (actual noise)            R_ekf (assumed R)
         ↓                                   ↓
    = 🟠 Measurements (orange dots)    Kalman Gain
         ↓ ─────────────────────────────→   ↓
             (EKF sees ONLY this!)      = Estimate (🟥 red)
    ```
    
    **NEW**: You can now see the measurements (🟠 orange dots) on the plots!
    - Change R_true → measurements scatter more/less around true state
    - EKF processes these noisy measurements to produce smooth estimate
    
    ### Parameter Guide:
    
    | Parameter | What It Controls | Visible On Plot |
    |-----------|------------------|-----------------|
    | **Q_true** | How much true state wiggles | Gray/black line variation |
    | **Q_ekf** | How fast EKF adapts | Red line responsiveness |
    | **R_true** | Actual sensor noise | 🟠 Orange dots scatter |
    | **R_ekf** | How much EKF trusts sensor | Red line smoothness |
    
    ### The Mismatch Scenario:
    
    **What happens when Q_ekf ≠ Q_true?**
    - If Q_ekf < Q_true: EKF is overconfident → poor tracking
    - If Q_ekf > Q_true: EKF is too cautious → noisy estimates
    
    **What happens when R_ekf ≠ R_true?**
    - If R_ekf < R_true: EKF trusts bad sensor too much → bias
    - If R_ekf > R_true: EKF ignores good sensor → misses info
    
    ### Try This Experiment:
    
    1. Set **Random Seed = 42** (keep it fixed!)
    2. Set True Q_Ca = -3 (moderate disturbances)
    3. Set EKF Q_Ca = -5 (EKF thinks disturbances are tiny)
    4. Run → Watch: Red line won't track gray line well (overconfident)
    5. Now change **only** EKF Q_Ca = -3 (matched!)
    6. Run → Watch: Much better tracking!
    
    Because the seed is fixed, the gray line (true state) is identical in both runs.
    You're seeing the **pure effect** of EKF tuning!
    
    **This is why EKF tuning matters in real applications!**
    """)
    

# --- Sidebar Controls ---
st.sidebar.header("Simulation Settings")

# Nonlinearity mode selection
nonlinearity_mode = st.sidebar.selectbox(
    "Reactor Nonlinearity",
    ["Standard CSTR", "Highly Nonlinear CSTR"],
    help="Highly Nonlinear mode adds: autocatalytic reaction, substrate inhibition, and thermal runaway potential"
)

sim_time = st.sidebar.slider("Simulation Time (s)", 5.0, 60.0, 10.0, step=1.0)
dt = 0.1
steps = int(sim_time / dt)

# Random seed for reproducibility
st.sidebar.markdown("**Random Seed** (for reproducibility)")
random_seed = st.sidebar.number_input("Seed", 0, 9999, 42, step=1,
                                      help="Set seed for reproducible noise. Same seed = same true state trajectory.",
                                      label_visibility="collapsed")
st.sidebar.caption("🎲 Same seed → Same noise → Compare EKF tuning fairly!")
np.random.seed(random_seed)

st.sidebar.header("Reactor Parameters")

# Set defaults based on mode
if nonlinearity_mode == "Highly Nonlinear CSTR":
    default_dH = -150000.0  # Much more exothermic
    default_E = 80000.0     # Higher activation energy
    default_k0_exp = 10.0   # Higher pre-exponential
    default_UA = 20000.0    # Less cooling (easier runaway)
else:
    default_dH = -50000.0
    default_E = 50000.0
    default_k0_exp = 6.86
    default_UA = 50000.0

col_a, col_b = st.sidebar.columns(2)
with col_a:
    q = st.number_input("Flow rate q (L/min)", 1.0, 500.0, 3.0, step=1.0)
    V = st.number_input("Volume V (L)", 1.0, 500.0, 10.0, step=10.0)
    Cai = st.number_input("Inlet Conc Cai (mol/L)", 0.1, 10.0, 1.0, step=0.1)
with col_b:
    Ti = st.number_input("Inlet Temp Ti (K)", 300.0, 400.0, 350.0, step=5.0)
    UA = st.number_input("Heat Transfer UA (W/K)", 1000.0, 200000.0, default_UA, step=5000.0)

st.sidebar.header("Reaction Kinetics")
col_c, col_d = st.sidebar.columns(2)
with col_c:
    dH = st.number_input("Enthalpy ΔH (J/mol)", -500000.0, 0.0, default_dH, step=10000.0)
    E = st.number_input("Activation E (J/mol)", 10000.0, 150000.0, default_E, step=5000.0)
with col_d:
    k0_exp = st.number_input("k₀ exponent (10^x)", 3.0, 12.0, default_k0_exp, step=0.5)
    k0 = 10**k0_exp

# Highly nonlinear mode parameters
if nonlinearity_mode == "Highly Nonlinear CSTR":
    st.sidebar.subheader("Additional Nonlinear Terms")
    k2 = st.sidebar.number_input("2nd order rate k₂ (L/(mol·min))", 0.0, 50.0, 10.0, step=1.0,
                                   help="Autocatalytic/second-order reaction")
    Ki = st.sidebar.number_input("Inhibition const Kᵢ (mol/L)", 0.01, 10.0, 0.5, step=0.1,
                                   help="Substrate inhibition constant")
    dH2 = st.sidebar.number_input("ΔH₂ for 2nd rxn (J/mol)", -300000.0, 0.0, -100000.0, step=10000.0)
else:
    k2 = 0.0
    Ki = 1e10  # Very large = no inhibition
    dH2 = 0.0

st.sidebar.header("Coolant Disturbance Profile")
st.sidebar.info("Coolant variations are the primary disturbance")

coolant_profile = st.sidebar.selectbox(
    "Disturbance Type",
    ["Step Change", "Sinusoidal", "Random Walk", "Multiple Steps", "Ramp", "Constant"],
    help="How coolant temperature varies over time"
)

if coolant_profile == "Step Change":
    tc_initial = st.sidebar.number_input("Initial Tc (K)", 280.0, 400.0, 300.0)
    tc_final = st.sidebar.number_input("Final Tc (K)", 280.0, 450.0, 350.0)
    tc_step_time = st.sidebar.slider("Step Time (s)", 1.0, sim_time-1, 5.0)
    
elif coolant_profile == "Sinusoidal":
    tc_mean = st.sidebar.number_input("Mean Tc (K)", 280.0, 400.0, 325.0)
    tc_amplitude = st.sidebar.slider("Amplitude (K)", 5.0, 50.0, 25.0)
    tc_period = st.sidebar.slider("Period (s)", 2.0, sim_time, 8.0)
    
elif coolant_profile == "Random Walk":
    tc_start = st.sidebar.number_input("Start Tc (K)", 280.0, 400.0, 300.0)
    tc_volatility = st.sidebar.slider("Volatility", 0.1, 5.0, 1.0,
                                      help="How much Tc changes per timestep")
    tc_min = st.sidebar.number_input("Min Tc (K)", 280.0, 350.0, 290.0)
    tc_max = st.sidebar.number_input("Max Tc (K)", 320.0, 450.0, 360.0)
    
elif coolant_profile == "Multiple Steps":
    tc_base = st.sidebar.number_input("Base Tc (K)", 280.0, 400.0, 300.0)
    tc_step_size = st.sidebar.slider("Step Size (K)", 5.0, 50.0, 20.0)
    tc_step_interval = st.sidebar.slider("Step Interval (s)", 1.0, 10.0, 3.0)
    
elif coolant_profile == "Ramp":
    tc_start = st.sidebar.number_input("Start Tc (K)", 280.0, 400.0, 300.0)
    tc_end = st.sidebar.number_input("End Tc (K)", 280.0, 450.0, 360.0)
    
elif coolant_profile == "Constant":
    tc_constant = st.sidebar.number_input("Coolant Tc (K)", 280.0, 400.0, 320.0)

st.sidebar.header("True System Noise")
st.sidebar.info("⬛ **Physical Reality** - Affects true state (black/gray line)")
q_true_ca = st.sidebar.slider("True Q_Ca", -8.0, -1.0, -4.0, format="1e%.0f",
                               help="Actual process noise in reactor (physical reality)",
                               key="q_true_ca")
q_true_t = st.sidebar.slider("True Q_T", -4.0, 2.0, 0.0, format="1e%.0f",
                              help="Actual temperature disturbances",
                              key="q_true_t")
r_true_t = st.sidebar.slider("True R_T", -2.0, 2.0, 0.3, format="1e%.1f",
                              help="Actual sensor noise (physical reality)",
                              key="r_true_t")
st.sidebar.caption("These parameters affect what actually happens in the reactor. Change these to see true state wiggle more/less.")

st.sidebar.header("Initial Conditions")
col_init1, col_init2 = st.sidebar.columns(2)
with col_init1:
    st.markdown("**True State**")
    Ca_init = st.number_input("Initial Ca (mol/L)", 0.0, 2.0, 0.5, step=0.1, key="ca_true")
    T_init = st.number_input("Initial T (K)", 300.0, 450.0, 350.0, step=5.0, key="t_true")
with col_init2:
    st.markdown("**EKF Estimate**")
    Ca_est_init = st.number_input("Est. Ca (mol/L)", 0.0, 2.0, 0.5, step=0.1, key="ca_est",
                                   help="Start with wrong estimate to see EKF convergence")
    T_est_init = st.number_input("Est. T (K)", 300.0, 450.0, 350.0, step=5.0, key="t_est")

st.sidebar.header("EKF Tuning")
st.sidebar.success("🟥 **EKF's Beliefs** - Affects estimate (red line) ONLY")
st.sidebar.caption("These are tuning parameters. They tell the EKF what to assume about noise, but don't change physical reality!")

# Tuning presets
tuning_preset = st.sidebar.selectbox(
    "Tuning Preset",
    ["Custom", "Good Tracking (Temp Only)", "Conservative", "Aggressive"],
    help="Pre-configured EKF tuning (does NOT change true system noise)"
)

if tuning_preset == "Good Tracking (Temp Only)":
    default_q_ca, default_q_t, default_r_t = -2.5, 0.0, 0.0
elif tuning_preset == "Conservative":
    default_q_ca, default_q_t, default_r_t = -4.0, -1.0, 0.5
elif tuning_preset == "Aggressive":
    default_q_ca, default_q_t, default_r_t = -1.0, 1.0, -0.5
else:  # Custom
    default_q_ca, default_q_t, default_r_t = -4.0, 0.0, 0.3

st.sidebar.markdown("**Process Noise Q** (EKF assumption)")
q_ca = st.sidebar.slider("EKF Q_Ca", -8.0, -1.0, default_q_ca, format="1e%.0f",
                          help="What EKF assumes Q is. Higher = faster adaptation.",
                          key="q_ekf_ca")
q_t = st.sidebar.slider("EKF Q_T", -4.0, 2.0, default_q_t, format="1e%.0f",
                         help="What EKF assumes Q_T is",
                         key="q_ekf_t")

st.sidebar.markdown("**Measurement Noise R** (EKF assumption)")
r_ca = st.sidebar.slider("EKF R_Ca", -6.0, 0.0, -3.7, format="1e%.1f",
                          help="What EKF assumes R_Ca is",
                          key="r_ekf_ca")
r_t = st.sidebar.slider("EKF R_T", -2.0, 2.0, default_r_t, format="1e%.1f",
                         help="What EKF assumes R_T is. Higher = trust sensor less.",
                         key="r_ekf_t")

if tuning_preset == "Good Tracking (Temp Only)":
    st.sidebar.info("""
    **Good Tracking preset:**
    - Q_Ca = 10^-2.5 ≈ 0.003 (moderate Ca uncertainty)
    - Q_T = 1.0 (moderate T uncertainty)
    - R_T = 1.0 (moderate sensor noise)
    
    Allows EKF to adapt Ca estimate based on T measurements via cross-covariance.
    """)

st.sidebar.header("Measurement Mode")
meas_mode = st.sidebar.radio("What can we measure?", ["Temperature Only", "Both Ca and T"])

st.sidebar.header("Advanced Options")
show_pure_model = st.sidebar.checkbox("Show Pure Model (zero-noise physics)", value=False,
                                      help="Display the deterministic ODE solution with no process noise. Useful for understanding the underlying physics.")

# Fixed constants
rho = 1000.0    # Density (g/L)
Cp = 4.184      # Heat capacity (J/(g·K))
R_gas = 8.314   # Gas constant (J/(mol·K))

# Create CSTR model parameters dictionary
cstr_params = {
    'q': q, 'V': V, 'Cai': Cai, 'Ti': Ti,
    'k0': k0, 'E': E, 'dH': dH,
    'UA': UA,
    'rho': rho, 'Cp': Cp, 'R_gas': R_gas,
    'k2': k2, 'Ki': Ki, 'dH2': dH2,
    'nonlinear_mode': (nonlinearity_mode == "Highly Nonlinear CSTR")
}

# Initialize CSTR model
cstr_model = CSTRModel(cstr_params)

# Build noise matrices - SEPARATE for true system vs EKF!
# True system noise (physical reality)
Q_true = np.diag([10**q_true_ca, 10**q_true_t])
R_true_T = 10**r_true_t
R_true_Ca = 10**-3.7  # Default, only used if measuring both

# EKF's belief about noise (tuning parameters)
Q_ekf = np.diag([10**q_ca, 10**q_t])
R_ekf_Ca = 10**r_ca
R_ekf_T = 10**r_t

# ==========================================
# 3. GENERATE COOLANT PROFILE
# ==========================================
# Build coolant profile parameters
coolant_params = {
    'tc_initial': tc_initial if coolant_profile == "Step Change" else 300.0,
    'tc_final': tc_final if coolant_profile == "Step Change" else 350.0,
    'tc_step_time': tc_step_time if coolant_profile == "Step Change" else 5.0,
    'tc_mean': tc_mean if coolant_profile == "Sinusoidal" else 325.0,
    'tc_amplitude': tc_amplitude if coolant_profile == "Sinusoidal" else 25.0,
    'tc_period': tc_period if coolant_profile == "Sinusoidal" else 8.0,
    'tc_start': tc_start if coolant_profile in ["Random Walk", "Ramp"] else 300.0,
    'tc_volatility': tc_volatility if coolant_profile == "Random Walk" else 1.0,
    'tc_min': tc_min if coolant_profile == "Random Walk" else 290.0,
    'tc_max': tc_max if coolant_profile == "Random Walk" else 360.0,
    'tc_base': tc_base if coolant_profile == "Multiple Steps" else 300.0,
    'tc_step_size': tc_step_size if coolant_profile == "Multiple Steps" else 20.0,
    'tc_step_interval': tc_step_interval if coolant_profile == "Multiple Steps" else 3.0,
    'tc_end': tc_end if coolant_profile == "Ramp" else 360.0,
    'tc_constant': tc_constant if coolant_profile == "Constant" else 320.0,
    'sim_time': sim_time
}

# ==========================================
# 4. SIMULATION LOOP
# ==========================================
t_span = np.linspace(0, sim_time, steps)

# Generate coolant profile using utility function
Tc_profile = generate_coolant_profile(coolant_profile, t_span, coolant_params, dt)

# Initial Conditions (from sidebar)
x_true = np.array([Ca_init, T_init])  # True state

# Pure model (optional) - uses true initial conditions
if show_pure_model:
    x_model = np.array([Ca_init, T_init])

# Storage
history = {
    't': [], 'Tc': [],
    'Ca_true': [], 'T_true': [],
    'Ca_meas': [], 'T_meas': [],  # Sensor readings (with noise)
    'Ca_est': [], 'T_est': [],
    'K_Ca': [], 'P_Ca': [], 'P_T': [], 'P_12': [],
    'k_rate': [],       # Reaction rate
    'Q_flow': [],       # Heat from flow (K/min)
    'Q_rxn': [],        # Heat from reaction (K/min)
    'Q_cool': []        # Heat from cooling (K/min)
}

if show_pure_model:
    history['Ca_model'] = []
    history['T_model'] = []

# Measurement setup - EKF uses its belief about R
if meas_mode == "Both Ca and T":
    H = np.eye(2)
    R_ekf_mat = np.diag([R_ekf_Ca, R_ekf_T])  # EKF's belief
else:
    H = np.array([[0, 1]])
    R_ekf_mat = np.array([[R_ekf_T]])  # EKF's belief

# Initialize EKF
P0 = np.diag([0.1, 100.0])
ekf = ExtendedKalmanFilter(
    model=cstr_model,
    H=H,
    Q=Q_ekf,
    R=R_ekf_mat,
    P0=P0,
    x0=np.array([Ca_est_init, T_est_init])
)

# Run Simulation
for k in range(steps):
    t = k * dt
    
    # Coolant temperature from profile
    Tc = Tc_profile[k]
    
    # A. PURE MODEL (no noise reference) - OPTIONAL
    if show_pure_model:
        sol_model = solve_ivp(lambda t, x: cstr_model.dynamics(t, x, Tc), [0, dt], x_model, 
                              method='RK45', rtol=1e-6, atol=1e-8)
        x_model = sol_model.y[:, -1]
        x_model[0] = max(0.0, x_model[0])
        x_model[1] = max(273.0, x_model[1])
    
    # B. TRUE SYSTEM (with TRUE process noise - physical reality)
    sol_true = solve_ivp(lambda t, x: cstr_model.dynamics(t, x, Tc), [0, dt], x_true,
                         method='RK45', rtol=1e-6, atol=1e-8)
    x_true = sol_true.y[:, -1]
    x_true = x_true + np.random.multivariate_normal([0, 0], Q_true)  # Use Q_true!
    x_true[0] = max(0.0, x_true[0])
    x_true[1] = max(273.0, x_true[1])
    
    # C. GENERATE MEASUREMENTS (with TRUE measurement noise - physical reality)
    if meas_mode == "Both Ca and T":
        y_meas = np.array([
            x_true[0] + np.random.normal(0, np.sqrt(R_true_Ca)),  # Use R_true!
            x_true[1] + np.random.normal(0, np.sqrt(R_true_T))    # Use R_true!
        ])
    else:
        y_meas = np.array([x_true[1] + np.random.normal(0, np.sqrt(R_true_T))])  # Use R_true!
    
    # D. EKF PREDICT + UPDATE (using modular EKF class)
    # Reference: Welch & Bishop (2006) "An Introduction to the Kalman Filter"
    # Reference: Dan Simon (2006) "Optimal State Estimation"
    x_est = ekf.step(y_meas, Tc, dt)
    P = ekf.get_covariance()
    K = ekf.get_kalman_gain()
    
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
    
    # Store measurements (what sensor actually reads)
    if meas_mode == "Both Ca and T":
        history['Ca_meas'].append(y_meas[0])
        history['T_meas'].append(y_meas[1])
    else:
        history['Ca_meas'].append(np.nan)  # Not measured
        history['T_meas'].append(y_meas[0])
    
    if show_pure_model:
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

# Chart styling and colors imported from utils

# ==========================================
# 6. VISUALIZATION
# ==========================================

# Show simulation info
st.caption(f"🎲 Random Seed: {random_seed} | 📊 {steps} time steps | ⏱️ {sim_time}s simulation")

# Display info about nonlinearity mode
if nonlinearity_mode == "Highly Nonlinear CSTR":
    st.info(f"""
    **Highly Nonlinear Mode Active** - This CSTR includes extreme nonlinearities:
    
    1. **Substrate Inhibition**: Rate = k·Ca / (1 + Ca/Kᵢ) - reaction slows at high concentration
    2. **Autocatalytic Reaction**: r₂ = k₂·Ca² - second-order kinetics create strong nonlinearity  
    3. **Multiple Exothermic Reactions**: ΔH₁ = {dH/1000:.0f} kJ/mol, ΔH₂ = {dH2/1000:.0f} kJ/mol - amplified thermal coupling
    4. **Thermal Runaway Potential**: High E ({E/1000:.0f} kJ/mol) + low cooling (UA = {UA:.0f} W/K) - small changes can trigger large excursions
    
    **EKF Challenge**: The linearization (Jacobian) is less accurate for such extreme nonlinearities. Watch how the EKF struggles or succeeds!
    """)
else:
    st.success("""
    **Standard CSTR** - Single exothermic reaction with Arrhenius kinetics. Moderately nonlinear due to exponential temperature dependence.
    """)

# Show coolant profile info
if coolant_profile != "Constant":
    st.info(f"""
    **Coolant Disturbance:** {coolant_profile} - Watch how the EKF tracks concentration as coolant temperature varies!
    """)

# Refresh colors based on current theme (in case theme changed)
COLORS = get_colors()

tab1, tab2, tab3 = st.tabs(["State Estimates", "EKF Internals", "Reaction Dynamics"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Concentration")
        fig_ca = go.Figure()
        
        # True state - Economist grey
        ca_label = 'True state' if meas_mode == "Both Ca and T" else 'True state (unmeasured)'
        line_style = dict(color=COLORS['true_state'], width=1.5, dash='dash') if meas_mode == "Temperature Only" else dict(color=COLORS['true_state'], width=2)
        fig_ca.add_trace(go.Scatter(x=df['t'], y=df['Ca_true'], mode='lines',
                                    name=ca_label, line=line_style))
        
        # Pure Model (optional)
        if show_pure_model:
            fig_ca.add_trace(go.Scatter(x=df['t'], y=df['Ca_model'], mode='lines',
                                        name='Model (no noise)', line=dict(color=COLORS['model'], width=1.5, dash='dot')))
        
        # Measurements - orange dots
        if meas_mode == "Both Ca and T":
            fig_ca.add_trace(go.Scatter(x=df['t'], y=df['Ca_meas'], mode='markers',
                                        name='Sensor readings', 
                                        marker=dict(color=COLORS['measurements'], size=6, opacity=0.7,
                                                   line=dict(color='black', width=1)),
                                        showlegend=True))
        
        fig_ca.add_trace(go.Scatter(x=df['t'], y=df['Ca_est'], mode='lines',
                                    name='EKF estimate', line=dict(color=COLORS['estimate'], width=2)))
        
        # Confidence interval - subtle
        sigma = np.sqrt(df['P_Ca'])
        fig_ca.add_trace(go.Scatter(x=df['t'], y=df['Ca_est'] + 2*sigma, mode='lines',
                                    line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_ca.add_trace(go.Scatter(x=df['t'], y=df['Ca_est'] - 2*sigma, mode='lines',
                                    fill='tonexty', fillcolor=COLORS['ci_fill'], 
                                    line=dict(width=0), name='95% uncertainty', hoverinfo='skip'))
        
        # Apply Economist styling
        fig_ca = economist_chart_style(fig_ca, height=350)
        fig_ca.update_xaxes(title_text="Time (seconds)")
        fig_ca.update_yaxes(title_text="Concentration (mol/L)")
        
        st.plotly_chart(fig_ca, use_container_width=True)
        
        if meas_mode == "Temperature Only":
            st.info("""
            **What You're Seeing vs. Reality:**
            
            - **Gray dashed (True State)**: Simulation-only reference - you would NOT have this in reality! 
              No Ca sensor means you never know the actual concentration.
            
            - **Orange dots (T Measurements)**: What temperature sensor ACTUALLY reads = True + R_true noise.
              This is the only raw data available!
            
            - **Red line (EKF Estimate)**: What you WOULD have - your best estimate based on 
              noisy measurements, physics model, and cross-covariance P₁₂.
            
            - **Shaded area (95% CI)**: Your uncertainty - true value should be within this ~95% of the time.
            
            **Signal Flow**: True State → Sensor (adds R_true) → Orange Measurements → EKF → Red Estimate
            
            **In a real reactor**: You'd have orange dots + red line. The gray dashed line doesn't exist!
            """)
    
    with col2:
        st.markdown("### Temperature")
        fig_t = go.Figure()
        
        # True state
        fig_t.add_trace(go.Scatter(x=df['t'], y=df['T_true'], mode='lines',
                                   name='True state', line=dict(color=COLORS['true_state'], width=2)))
        
        # Pure Model (optional)
        if show_pure_model:
            fig_t.add_trace(go.Scatter(x=df['t'], y=df['T_model'], mode='lines',
                                       name='Model (no noise)', line=dict(color=COLORS['model'], width=1.5, dash='dot')))
        
        # Measurements - orange dots
        fig_t.add_trace(go.Scatter(x=df['t'], y=df['T_meas'], mode='markers',
                                   name='Sensor readings', 
                                   marker=dict(color=COLORS['measurements'], size=6, opacity=0.7,
                                              line=dict(color='black', width=1)),
                                   showlegend=True))
        
        fig_t.add_trace(go.Scatter(x=df['t'], y=df['T_est'], mode='lines',
                                   name='EKF estimate', line=dict(color=COLORS['estimate'], width=2)))
        fig_t.add_trace(go.Scatter(x=df['t'], y=df['Tc'], mode='lines',
                                   name='Coolant input', line=dict(color=COLORS['coolant'], width=1.8, dash='dot')))
        
        # Apply Economist styling
        fig_t = economist_chart_style(fig_t, height=350)
        fig_t.update_xaxes(title_text="Time (seconds)")
        fig_t.update_yaxes(title_text="Temperature (K)")
        
        st.plotly_chart(fig_t, use_container_width=True)

with tab2:
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Kalman Gain (K_Ca)")
        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(x=df['t'], y=df['K_Ca'], mode='lines',
                                   line=dict(color='green', width=2)))
        if coolant_profile == "Step Change":
            fig_k.add_vline(x=tc_step_time, line_dash="dot", line_color="gray")
        fig_k = economist_chart_style(fig_k, height=350)
        fig_k.update_xaxes(title_text="Time (s)")
        fig_k.update_yaxes(title_text="K_Ca")
        st.plotly_chart(fig_k, width='stretch')
        st.caption("How much the filter trusts measurements for concentration updates. K→0: trust model, K→1: trust sensor.")
    
    with col4:
        st.subheader("Cross-Covariance P12")
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=df['t'], y=df['P_12'], mode='lines',
                                   line=dict(color='purple', width=2)))
        if coolant_profile == "Step Change":
            fig_p.add_vline(x=tc_step_time, line_dash="dot", line_color="gray")
        fig_p = economist_chart_style(fig_p, height=350)
        fig_p.update_xaxes(title_text="Time (s)")
        fig_p.update_yaxes(title_text="P₁₂")
        st.plotly_chart(fig_p, width='stretch')
        st.caption("Coupling between Ca and T estimates. Negative = inverse relationship (higher T → lower Ca).")

with tab3:
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("Reaction Rate (Arrhenius)")
        fig_rate = go.Figure()
        fig_rate.add_trace(go.Scatter(x=df['t'], y=df['k_rate'], mode='lines',
                                      name='k (1/min)', line=dict(color='orange', width=2)))
        if coolant_profile == "Step Change":
            fig_rate.add_vline(x=tc_step_time, line_dash="dot", line_color="gray")
        fig_rate = economist_chart_style(fig_rate, height=350)
        fig_rate.update_xaxes(title_text="Time (s)")
        fig_rate.update_yaxes(title_text="k (1/min)")
        st.plotly_chart(fig_rate, width='stretch')
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
        if coolant_profile == "Step Change":
            fig_heat.add_vline(x=tc_step_time, line_dash="dot", line_color="gray")
        fig_heat = economist_chart_style(fig_heat, height=350)
        fig_heat.update_layout(
            margin=dict(l=0, r=130, t=30, b=0),
            legend=dict(
                orientation="v",
                yanchor="top", y=1.0,
                xanchor="left", x=1.02,
                bgcolor=COLORS['legend_bg'],
                bordercolor=COLORS['legend_border'],
                borderwidth=1,
                font=dict(color=COLORS['text'])
            )
        )
        fig_heat.update_xaxes(title_text="Time (s)")
        fig_heat.update_yaxes(title_text="Heat Rate (K/min)")
        st.plotly_chart(fig_heat, width='stretch')
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
