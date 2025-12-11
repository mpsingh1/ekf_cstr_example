import streamlit as st

st.set_page_config(page_title="EKF Mathematics", layout="wide")

st.title("Extended Kalman Filter Mathematics")
st.markdown("""
This page explains the mathematical foundation of the EKF implementation.

**Primary References:**
1. Welch, G., & Bishop, G. (2006). *An Introduction to the Kalman Filter*. Technical Report TR 95-041, University of North Carolina.
2. Simon, D. (2006). *Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches*. Wiley-Interscience.
3. Jazwinski, A. H. (1970). *Stochastic Processes and Filtering Theory*. Academic Press.
""")

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
st.caption("First-order Euler approximation of the matrix exponential exp(A·Δt)")

st.markdown("### Jacobian Derivation")
st.markdown("""
Given the CSTR dynamics:
- $f_1 = \\frac{q}{V}(C_{Ai} - C_A) - kC_A$
- $f_2 = \\frac{q}{V}(T_i - T) + \\frac{(-\\Delta H)}{\\rho C_p}kC_A + \\frac{UA}{V\\rho C_p}(T_c - T)$

Where $k = k_0 e^{-E/(RT)}$, we have:
""")

st.latex(r"\frac{\partial k}{\partial T} = k \cdot \frac{E}{RT^2}")

st.markdown("Therefore:")
st.latex(r"\frac{\partial f_1}{\partial C_A} = -\frac{q}{V} - k, \quad \frac{\partial f_1}{\partial T} = -C_A \frac{\partial k}{\partial T} = -C_A k \frac{E}{RT^2}")
st.latex(r"\frac{\partial f_2}{\partial C_A} = \frac{(-\Delta H)}{\rho C_p}k, \quad \frac{\partial f_2}{\partial T} = -\frac{q}{V} + \frac{(-\Delta H)}{\rho C_p}C_A\frac{\partial k}{\partial T} - \frac{UA}{V\rho C_p}")

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
    
    st.markdown("**Covariance Update (Joseph Form):**")
    st.latex(r"\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k\mathbf{H})\mathbf{P}_{k|k-1}(\mathbf{I} - \mathbf{K}_k\mathbf{H})^T + \mathbf{K}_k\mathbf{R}\mathbf{K}_k^T")
    
    st.info("""
    **Note on Joseph Form:** The standard form is $\\mathbf{P} = (\\mathbf{I} - \\mathbf{K}\\mathbf{H})\\mathbf{P}_{pred}$, 
    but the Joseph form (shown above) is preferred for numerical stability. It guarantees positive-definiteness 
    of $\\mathbf{P}$ even with rounding errors (Bucy & Joseph, 1968; Simon, 2006, Section 5.3).
    """)

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
    st.markdown("""
    **All states** follow model prediction.
    
    Happens when: $\mathbf{R}$ large or $\mathbf{Q}$ small
    """)

with col13:
    st.markdown("### $\mathbf{K} \\to \\mathbf{H}^+$ (pseudoinverse)")
    st.latex(r"\hat{\mathbf{x}}_{\text{measured}} \approx \mathbf{y}")
    st.success("**Trust the SENSOR** (ignore model)")
    st.markdown("""
    **Measured states** follow sensor directly.
    
    **Unmeasured states** still depend on model + cross-covariance.
    
    Happens when: $\mathbf{R}$ small or $\mathbf{Q}$ large
    """)

st.info("""
**Important for Partial Measurements:**

When measuring only **some** states (e.g., only Temperature in CSTR):
- $\\mathbf{K}$ is a 2×1 vector: $\\mathbf{K} = [K_{C_A}, K_T]^T$
- $K_T$ controls how much temperature measurement affects the temperature estimate
- $K_{C_A}$ controls how much temperature measurement affects the **unmeasured** concentration estimate
- The unmeasured state (Ca) is inferred through **cross-covariance** $P_{12}$, not direct measurement
- You can **never** fully "trust the sensor" for unmeasured states - you're always relying on the model's correlation structure
""")

st.divider()

# ===========================================
# 7. CROSS-COVARIANCE AND UNMEASURED STATES
# ===========================================
st.header("7. Cross-Covariance and Unmeasured States")

st.markdown("""
The off-diagonal element $P_{12}$ of the covariance matrix captures the **correlation** between state estimates:
""")

st.latex(r"\mathbf{P} = \begin{bmatrix} P_{11} & P_{12} \\ P_{21} & P_{22} \end{bmatrix} = \begin{bmatrix} \text{Var}(C_A) & \text{Cov}(C_A, T) \\ \text{Cov}(T, C_A) & \text{Var}(T) \end{bmatrix}")

st.markdown("### How Unmeasured States Are Updated")

st.markdown("""
When measuring **only temperature** (H = [0, 1]), the Kalman Gain becomes:
""")

st.latex(r"\mathbf{K} = \begin{bmatrix} K_{C_A} \\ K_T \end{bmatrix} = \mathbf{P}_{pred} \mathbf{H}^T (\mathbf{H} \mathbf{P}_{pred} \mathbf{H}^T + R)^{-1}")

st.markdown("""
Expanding this for partial measurements:
""")

st.latex(r"K_{C_A} = \frac{P_{12}}{P_{22} + R_T}, \quad K_T = \frac{P_{22}}{P_{22} + R_T}")

st.markdown("""
The key insight: **$K_{C_A}$ depends on $P_{12}$**, the cross-covariance!

The state update becomes:
""")

st.latex(r"\hat{C}_A = \hat{C}_{A,pred} + K_{C_A}(T_{measured} - \hat{T}_{pred})")
st.latex(r"\hat{T} = \hat{T}_{pred} + K_T(T_{measured} - \hat{T}_{pred})")

st.markdown("""
**Physical Interpretation for CSTR:**
- When $P_{12} < 0$: Temperature and concentration are **negatively correlated**
- Meaning: If temperature increases unexpectedly (positive innovation), concentration likely decreased
- **Why?** Exothermic reaction: more reaction → consumes reactant (Ca ↓) → releases heat (T ↑)
- The EKF exploits this coupling to infer unmeasured Ca from measured T

**Bottom Line:** You cannot estimate unmeasured states without:
1. A good **model** (to predict their evolution)
2. **Cross-covariance** $P_{12}$ (to link them to measured states)
3. Sufficient **process noise** Q (to allow the filter to adapt)

This is the **"hidden link"** that allows the filter to estimate unmeasured states - but it's always model-based inference, never direct observation.
""")

st.divider()

# ===========================================
# 8. EXAMPLE: TEMPERATURE-ONLY MEASUREMENT
# ===========================================
st.header("8. Example: Temperature-Only Measurement")

st.markdown("""
Consider a scenario where we **only measure temperature** but want to estimate both Ca and T.
""")

col14, col15 = st.columns(2)

with col14:
    st.markdown("### Scenario")
    st.code("""
# Predicted state
Ca_pred = 0.50 mol/L
T_pred = 360 K

# Covariance (simplified)
P = [[0.01,  -0.5 ],
     [-0.5,   25.0]]
P₁₂ = -0.5 (negative correlation)

# Measurement
T_measured = 365 K  (5 K higher!)
R_T = 1.0 K²
""", language="python")

with col15:
    st.markdown("### Kalman Gain Calculation")
    st.latex(r"K_{C_A} = \frac{P_{12}}{P_{22} + R_T} = \frac{-0.5}{25 + 1} \approx -0.019")
    st.latex(r"K_T = \frac{P_{22}}{P_{22} + R_T} = \frac{25}{25 + 1} \approx 0.96")
    
    st.markdown("### State Update")
    st.latex(r"\text{Innovation} = 365 - 360 = 5 \text{ K}")
    st.latex(r"\hat{C}_A = 0.50 + (-0.019)(5) = 0.405 \text{ mol/L}")
    st.latex(r"\hat{T} = 360 + (0.96)(5) = 364.8 \text{ K}")

st.success("""
**Key Observation:** 
- Temperature measurement was 5 K **higher** than expected
- Filter inferred that Ca is **lower** (0.50 → 0.405 mol/L) even though we **didn't measure Ca**!
- Why? Because P₁₂ < 0 indicates: higher T correlates with lower Ca (exothermic reaction)
- The magnitude of the Ca adjustment depends on:
  - How strong the correlation is (|P₁₂|)
  - How certain we are about T (P₂₂)
  - How much we trust the sensor (R_T)
""")

st.divider()

# ===========================================
# 9. IMPLEMENTATION NOTES
# ===========================================
st.header("9. Implementation Notes")

st.markdown("""
### Numerical Stability

**Covariance Update:** We use the Joseph form rather than the simplified form:
- **Joseph form (used):** $\\mathbf{P} = (\\mathbf{I} - \\mathbf{K}\\mathbf{H})\\mathbf{P}_{pred}(\\mathbf{I} - \\mathbf{K}\\mathbf{H})^T + \\mathbf{K}\\mathbf{R}\\mathbf{K}^T$
- **Simplified form:** $\\mathbf{P} = (\\mathbf{I} - \\mathbf{K}\\mathbf{H})\\mathbf{P}_{pred}$

The Joseph form guarantees positive semi-definiteness even with numerical rounding errors.

### Discretization

The continuous Jacobian $\\mathbf{A}$ is discretized using first-order Euler:
$$\\mathbf{F} = \\mathbf{I} + \\mathbf{A}\\Delta t$$

For small time steps (Δt = 0.1 s), this is accurate. For larger steps, consider matrix exponential: $\\mathbf{F} = e^{\\mathbf{A}\\Delta t}$.

### Linearization Point

The Jacobian is evaluated at the **current estimate** $\\hat{\\mathbf{x}}_{k-1}$, not the prediction. This is standard EKF practice (Simon, 2006, Algorithm 13.1).
""")

st.divider()

st.markdown("""
---
### References

1. **Welch, G., & Bishop, G. (2006).** *An Introduction to the Kalman Filter.* Technical Report TR 95-041, University of North Carolina at Chapel Hill. [Available online]

2. **Simon, D. (2006).** *Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches.* Wiley-Interscience. ISBN: 978-0-471-70858-2.

3. **Jazwinski, A. H. (1970).** *Stochastic Processes and Filtering Theory.* Academic Press.

4. **Bucy, R. S., & Joseph, P. D. (1968).** *Filtering for Stochastic Processes with Applications to Guidance.* Interscience Publishers.

5. **Rawlings, J. B., & Mayne, D. Q. (2009).** *Model Predictive Control: Theory and Design.* Nob Hill Publishing.

### Verification

The implementation follows the standard EKF algorithm as presented in:
- Simon (2006), Algorithm 13.1 (Extended Kalman Filter)
- Welch & Bishop (2006), Section 3.3 (The Discrete Extended Kalman Filter)
""")

