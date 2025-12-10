import streamlit as st

st.set_page_config(page_title="EKF Mathematics", layout="wide")

st.title("Extended Kalman Filter Mathematics")
st.markdown("This page explains the math behind the EKF implementation in the simulator.")

# ===========================================
# 1. STATE SPACE MODEL
# ===========================================
st.header("1. State-Space Model")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Continuous-Time Nonlinear System")
    st.latex(r"\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u}) + \mathbf{w}")
    st.latex(r"\mathbf{y} = h(\mathbf{x}) + \mathbf{v}")
    
    st.markdown("""
    Where:
    - $\mathbf{x}$ = state vector
    - $\mathbf{u}$ = input (control)
    - $\mathbf{y}$ = measurement
    - $\mathbf{w} \sim \mathcal{N}(0, \mathbf{Q})$ = process noise
    - $\mathbf{v} \sim \mathcal{N}(0, \mathbf{R})$ = measurement noise
    """)

with col2:
    st.markdown("### CSTR State Vector")
    st.latex(r"\mathbf{x} = \begin{bmatrix} C_A \\ T \end{bmatrix}")
    
    st.markdown("### CSTR Dynamics $f(\mathbf{x}, u)$")
    st.latex(r"f(\mathbf{x}, T_c) = \begin{bmatrix} \frac{q}{V}(C_{Ai} - C_A) - kC_A \\ \frac{q}{V}(T_i - T) + \frac{(-\Delta H)}{\rho C_p}kC_A + \frac{UA}{V\rho C_p}(T_c - T) \end{bmatrix}")

st.divider()

# ===========================================
# 2. LINEARIZATION (JACOBIAN)
# ===========================================
st.header("2. Linearization (Jacobian)")

st.markdown("""
The EKF linearizes the nonlinear model around the current estimate using the **Jacobian matrix**.
""")

col3, col4 = st.columns(2)

with col3:
    st.markdown("### Continuous Jacobian $\mathbf{A}$")
    st.latex(r"\mathbf{A} = \frac{\partial f}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f_1}{\partial C_A} & \frac{\partial f_1}{\partial T} \\ \frac{\partial f_2}{\partial C_A} & \frac{\partial f_2}{\partial T} \end{bmatrix}")

with col4:
    st.markdown("### Jacobian Elements")
    st.latex(r"a_{11} = -\frac{q}{V} - k")
    st.latex(r"a_{12} = -C_A \frac{\partial k}{\partial T} = -C_A k \frac{E}{RT^2}")
    st.latex(r"a_{21} = \frac{(-\Delta H)}{\rho C_p} k")
    st.latex(r"a_{22} = -\frac{q}{V} + \frac{(-\Delta H)}{\rho C_p} C_A \frac{\partial k}{\partial T} - \frac{UA}{V\rho C_p}")

st.markdown("### Discretization")
st.latex(r"\mathbf{F} = \mathbf{I} + \mathbf{A} \cdot \Delta t")
st.caption("First-order Euler approximation of the matrix exponential")

st.divider()

# ===========================================
# 3. EKF ALGORITHM
# ===========================================
st.header("3. EKF Algorithm")

st.markdown("The EKF operates in two steps: **Predict** and **Update**.")

col5, col6 = st.columns(2)

with col5:
    st.markdown("### Predict Step (Time Update)")
    st.markdown("**State Prediction:**")
    st.latex(r"\hat{\mathbf{x}}_{k|k-1} = f(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_k)")
    
    st.markdown("**Covariance Prediction:**")
    st.latex(r"\mathbf{P}_{k|k-1} = \mathbf{F}_k \mathbf{P}_{k-1|k-1} \mathbf{F}_k^T + \mathbf{Q}")
    
    st.info("Uses physics model to propagate state and uncertainty forward in time.")

with col6:
    st.markdown("### Update Step (Measurement Update)")
    st.markdown("**Innovation (Residual):**")
    st.latex(r"\mathbf{r}_k = \mathbf{y}_k - \mathbf{H}\hat{\mathbf{x}}_{k|k-1}")
    
    st.markdown("**Innovation Covariance:**")
    st.latex(r"\mathbf{S}_k = \mathbf{H}\mathbf{P}_{k|k-1}\mathbf{H}^T + \mathbf{R}")
    
    st.markdown("**Kalman Gain:**")
    st.latex(r"\mathbf{K}_k = \mathbf{P}_{k|k-1}\mathbf{H}^T\mathbf{S}_k^{-1}")
    
    st.markdown("**State Update:**")
    st.latex(r"\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \mathbf{r}_k")
    
    st.markdown("**Covariance Update:**")
    st.latex(r"\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k\mathbf{H})\mathbf{P}_{k|k-1}")
    
    st.success("Corrects prediction using sensor measurement.")

st.divider()

# ===========================================
# 4. MEASUREMENT MATRIX
# ===========================================
st.header("4. Measurement Matrix $\mathbf{H}$")

col7, col8 = st.columns(2)

with col7:
    st.markdown("### Temperature Only")
    st.latex(r"\mathbf{H} = \begin{bmatrix} 0 & 1 \end{bmatrix}")
    st.markdown("We observe only $T$ (second state).")
    st.latex(r"\mathbf{y} = T + v")

with col8:
    st.markdown("### Both Ca and T")
    st.latex(r"\mathbf{H} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \mathbf{I}")
    st.markdown("We observe both states.")
    st.latex(r"\mathbf{y} = \begin{bmatrix} C_A + v_1 \\ T + v_2 \end{bmatrix}")

st.divider()

# ===========================================
# 5. TUNING PARAMETERS
# ===========================================
st.header("5. Tuning Parameters")

col9, col10, col11 = st.columns(3)

with col9:
    st.markdown("### Process Noise $\mathbf{Q}$")
    st.latex(r"\mathbf{Q} = \begin{bmatrix} Q_{C_A} & 0 \\ 0 & Q_T \end{bmatrix}")
    st.markdown("""
    **Meaning:** Model uncertainty
    
    **Effect of increasing:**
    - More trust in measurements
    - Faster response
    - Noisier estimates
    """)

with col10:
    st.markdown("### Measurement Noise $\mathbf{R}$")
    st.latex(r"\mathbf{R} = \begin{bmatrix} R_{C_A} & 0 \\ 0 & R_T \end{bmatrix}")
    st.markdown("""
    **Meaning:** Sensor uncertainty
    
    **Effect of increasing:**
    - More trust in model
    - Slower response
    - Smoother estimates
    """)

with col11:
    st.markdown("### Initial Covariance $\mathbf{P}_0$")
    st.latex(r"\mathbf{P}_0 = \begin{bmatrix} P_{C_A,0} & 0 \\ 0 & P_{T,0} \end{bmatrix}")
    st.markdown("""
    **Meaning:** Initial uncertainty
    
    **Effect of increasing:**
    - Faster initial adaptation
    - Quick convergence
    """)

st.divider()

# ===========================================
# 6. KEY INSIGHT
# ===========================================
st.header("6. The Key Insight: Kalman Gain")

st.markdown("""
The **Kalman Gain** $\mathbf{K}$ optimally blends model prediction and sensor measurement:
""")

st.latex(r"\hat{\mathbf{x}} = \underbrace{\hat{\mathbf{x}}_{\text{pred}}}_{\text{Model}} + \mathbf{K} \cdot \underbrace{(\mathbf{y} - \mathbf{H}\hat{\mathbf{x}}_{\text{pred}})}_{\text{Innovation}}")

col12, col13 = st.columns(2)

with col12:
    st.markdown("### $\mathbf{K} \\to 0$")
    st.latex(r"\hat{\mathbf{x}} \approx \hat{\mathbf{x}}_{\text{pred}}")
    st.warning("**Trust the MODEL** (ignore sensor)")
    st.markdown("Happens when: $\mathbf{R}$ large or $\mathbf{Q}$ small")

with col13:
    st.markdown("### $\mathbf{K} \\to 1$")
    st.latex(r"\hat{\mathbf{x}} \approx \mathbf{y}")
    st.success("**Trust the SENSOR** (ignore model)")
    st.markdown("Happens when: $\mathbf{R}$ small or $\mathbf{Q}$ large")

st.divider()

# ===========================================
# 7. CROSS-COVARIANCE
# ===========================================
st.header("7. Cross-Covariance $P_{12}$")

st.markdown("""
The off-diagonal element $P_{12}$ of the covariance matrix captures the **correlation** between state estimates:
""")

st.latex(r"\mathbf{P} = \begin{bmatrix} P_{11} & P_{12} \\ P_{21} & P_{22} \end{bmatrix} = \begin{bmatrix} \text{Var}(C_A) & \text{Cov}(C_A, T) \\ \text{Cov}(T, C_A) & \text{Var}(T) \end{bmatrix}")

st.markdown("""
**Why it matters:**
- When $P_{12} < 0$: Temperature and concentration are **negatively correlated**
- This means: If $T$ increases, $C_A$ likely decreases (exothermic reaction consuming reactant)
- The EKF uses this coupling to update $C_A$ even when only measuring $T$!

This is the **"hidden link"** that allows the filter to estimate unmeasured states.
""")

st.divider()

st.markdown("""
---
### References
1. Welch, G., & Bishop, G. (2006). An Introduction to the Kalman Filter.
2. Simon, D. (2006). Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches.
3. Rawlings, J. B., & Mayne, D. Q. (2009). Model Predictive Control: Theory and Design.
""")

