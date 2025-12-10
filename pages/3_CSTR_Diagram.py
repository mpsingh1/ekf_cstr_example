import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="CSTR System Diagram", layout="wide")

st.title("Continuous Stirred Tank Reactor (CSTR)")
st.markdown("A visual guide to the reactor system being simulated.")

# ===========================================
# CSTR SCHEMATIC
# ===========================================
st.header("1. System Schematic")

# Create CSTR diagram using Plotly
fig = go.Figure()

# Reactor vessel (rectangle)
fig.add_shape(type="rect", x0=2, y0=1, x1=6, y1=5,
              line=dict(color="black", width=3),
              fillcolor="lightblue")

# Inlet pipe (top left)
fig.add_shape(type="line", x0=0, y0=4.5, x1=2, y1=4.5,
              line=dict(color="black", width=2))
fig.add_annotation(x=1, y=4.8, text="Feed In", showarrow=False, font=dict(size=12))
fig.add_annotation(x=1, y=4.2, text="q, CAi, Ti", showarrow=False, font=dict(size=10, color="blue"))

# Outlet pipe (bottom right)
fig.add_shape(type="line", x0=6, y0=1.5, x1=8, y1=1.5,
              line=dict(color="black", width=2))
fig.add_annotation(x=7, y=1.8, text="Product Out", showarrow=False, font=dict(size=12))
fig.add_annotation(x=7, y=1.2, text="q, CA, T", showarrow=False, font=dict(size=10, color="red"))

# Cooling jacket (outer rectangle)
fig.add_shape(type="rect", x0=1.5, y0=0.5, x1=6.5, y1=5.5,
              line=dict(color="green", width=2, dash="dash"),
              fillcolor="rgba(0,255,0,0.1)")

# Coolant inlet
fig.add_shape(type="line", x0=1.5, y0=0.8, x1=0.5, y1=0.8,
              line=dict(color="green", width=2))
fig.add_annotation(x=0.8, y=0.5, text="Coolant Tc", showarrow=False, font=dict(size=10, color="green"))

# Coolant outlet
fig.add_shape(type="line", x0=6.5, y0=5.2, x1=7.5, y1=5.2,
              line=dict(color="green", width=2))

# Stirrer
fig.add_shape(type="line", x0=4, y0=5.5, x1=4, y1=3,
              line=dict(color="gray", width=3))
fig.add_shape(type="line", x0=3.3, y0=2.5, x1=4.7, y1=2.5,
              line=dict(color="gray", width=3))
fig.add_shape(type="line", x0=3.5, y0=2.2, x1=4.5, y1=2.8,
              line=dict(color="gray", width=2))
fig.add_shape(type="line", x0=3.5, y0=2.8, x1=4.5, y1=2.2,
              line=dict(color="gray", width=2))

# Motor
fig.add_shape(type="rect", x0=3.5, y0=5.5, x1=4.5, y1=6.2,
              line=dict(color="gray", width=2),
              fillcolor="gray")
fig.add_annotation(x=4, y=6.5, text="Motor", showarrow=False, font=dict(size=10))

# Labels inside reactor
fig.add_annotation(x=4, y=3.5, text="V, CA, T", showarrow=False, 
                   font=dict(size=14, color="darkblue", family="Arial Black"))
fig.add_annotation(x=4, y=1.8, text="Reaction: A → B", showarrow=False, 
                   font=dict(size=12, color="red"))

# Temperature sensor
fig.add_shape(type="circle", x0=5.5, y0=3, x1=5.8, y1=3.3,
              line=dict(color="red", width=2),
              fillcolor="red")
fig.add_shape(type="line", x0=5.65, y0=3.3, x1=5.65, y1=4,
              line=dict(color="red", width=2))
fig.add_annotation(x=6.3, y=3.8, text="T sensor", showarrow=False, font=dict(size=10, color="red"))

fig.update_layout(
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 8.5]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 7]),
    height=500,
    margin=dict(l=20, r=20, t=20, b=20),
    plot_bgcolor='white'
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ===========================================
# COMPONENT DESCRIPTION
# ===========================================
st.header("2. Component Description")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Reactor Vessel")
    st.markdown("""
    - **Volume (V)**: The liquid holdup in the reactor
    - **Assumptions**: 
        - Perfectly mixed (uniform concentration and temperature)
        - Constant density and volume
        - Single liquid phase
    """)
    
    st.markdown("### Feed Stream")
    st.markdown("""
    - **Flow rate (q)**: Volumetric flow rate in and out (equal for constant volume)
    - **Inlet concentration (CAi)**: Fresh reactant concentration
    - **Inlet temperature (Ti)**: Feed stream temperature
    """)

with col2:
    st.markdown("### Cooling Jacket")
    st.markdown("""
    - **Coolant temperature (Tc)**: The manipulated variable
    - **Heat transfer (UA)**: Overall coefficient times area
    - **Purpose**: Remove heat from exothermic reaction
    """)
    
    st.markdown("### Instrumentation")
    st.markdown("""
    - **Temperature sensor**: Measures reactor temperature T
    - **Challenge**: Concentration CA is NOT directly measured
    - **Solution**: Use EKF to estimate CA from T
    """)

st.divider()

# ===========================================
# BALANCE EQUATIONS
# ===========================================
st.header("3. Balance Equations")

st.markdown("### Material Balance on Reactant A")
st.latex(r"\frac{d(VC_A)}{dt} = qC_{Ai} - qC_A - VkC_A")

st.markdown("For constant volume:")
st.latex(r"\frac{dC_A}{dt} = \frac{q}{V}(C_{Ai} - C_A) - kC_A")

col3, col4, col5 = st.columns(3)
with col3:
    st.markdown("**Accumulation**")
    st.latex(r"\frac{dC_A}{dt}")
with col4:
    st.markdown("**In - Out**")
    st.latex(r"\frac{q}{V}(C_{Ai} - C_A)")
with col5:
    st.markdown("**Consumption**")
    st.latex(r"-kC_A")

st.divider()

st.markdown("### Energy Balance")
st.latex(r"\frac{dT}{dt} = \frac{q}{V}(T_i - T) + \frac{(-\Delta H)}{\rho C_p}kC_A + \frac{UA}{V\rho C_p}(T_c - T)")

col6, col7, col8 = st.columns(3)
with col6:
    st.markdown("**Flow Heat**")
    st.latex(r"\frac{q}{V}(T_i - T)")
    st.caption("Heat from feed")
with col7:
    st.markdown("**Reaction Heat**")
    st.latex(r"\frac{(-\Delta H)}{\rho C_p}kC_A")
    st.caption("Exothermic release")
with col8:
    st.markdown("**Cooling Heat**")
    st.latex(r"\frac{UA}{V\rho C_p}(T_c - T)")
    st.caption("Jacket removal")

st.divider()

# ===========================================
# STATE SPACE FORM
# ===========================================
st.header("4. State-Space Representation")

st.markdown("The CSTR can be written in state-space form:")

st.latex(r"\dot{\mathbf{x}} = f(\mathbf{x}, u)")

st.markdown("Where:")
col9, col10 = st.columns(2)

with col9:
    st.markdown("**State Vector:**")
    st.latex(r"\mathbf{x} = \begin{bmatrix} C_A \\ T \end{bmatrix}")
    
with col10:
    st.markdown("**Input:**")
    st.latex(r"u = T_c")
    st.caption("Coolant temperature (manipulated)")

st.markdown("**Nonlinear Dynamics:**")
st.latex(r"f(\mathbf{x}, T_c) = \begin{bmatrix} \frac{q}{V}(C_{Ai} - C_A) - kC_A \\ \frac{q}{V}(T_i - T) + \frac{(-\Delta H)}{\rho C_p}kC_A + \frac{UA}{V\rho C_p}(T_c - T) \end{bmatrix}")

st.divider()

# ===========================================
# NONLINEARITY
# ===========================================
st.header("5. Source of Nonlinearity")

st.markdown("""
The CSTR is a **nonlinear system** due to the Arrhenius reaction rate:
""")

st.latex(r"k = k_0 \exp\left(-\frac{E}{RT}\right)")

st.markdown("""
This creates:
1. **Exponential dependence** of reaction rate on temperature
2. **Multiplicative coupling** between $k$ and $C_A$ in both balances
3. **Potential for multiple steady states** and thermal runaway

This nonlinearity is why we use the **Extended** Kalman Filter (EKF) rather than the standard linear Kalman Filter.
""")

st.info("""
**Thermal Runaway Risk:** If cooling is insufficient, higher T → faster reaction → more heat → even higher T → exponential temperature rise. The EKF helps monitor this by estimating both states.
""")

