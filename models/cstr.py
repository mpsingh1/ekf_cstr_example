"""
Continuous Stirred Tank Reactor (CSTR) Model

This module implements a nonlinear CSTR model with:
- Arrhenius kinetics
- Substrate inhibition (optional)
- Second-order autocatalytic reaction (optional)
- Energy balance with cooling
"""

import numpy as np


class CSTRModel:
    """CSTR Model for state estimation demonstrations"""
    
    def __init__(self, params):
        """
        Initialize CSTR model with parameters.
        
        Parameters:
        -----------
        params : dict
            Dictionary containing all model parameters:
            - q, V, Cai, Ti : Reactor parameters
            - k0, E, dH : Reaction kinetics
            - UA : Heat transfer
            - rho, Cp, R_gas : Thermodynamic constants
            - k2, Ki, dH2 : Additional nonlinear terms (optional)
            - nonlinear_mode : bool
        """
        self.params = params
        self.n_states = 2  # Ca, T
        
    def dynamics(self, t, x, u_Tc):
        """
        CSTR dynamics: dx/dt = f(x, u)
        
        Parameters:
        -----------
        t : float
            Time (not used in autonomous system)
        x : array_like, shape (2,)
            State vector [Ca, T]
        u_Tc : float
            Coolant temperature (control input)
            
        Returns:
        --------
        dxdt : list, shape (2,)
            State derivatives [dCa/dt, dT/dt]
        """
        p = self.params
        Ca, T = x
        Ca = max(0.0, Ca)
        T = max(273.0, T)
        
        # Primary Arrhenius rate law with substrate inhibition
        k_primary = p['k0'] * np.exp(-p['E'] / (p['R_gas'] * T))
        k_effective = k_primary / (1 + Ca / p['Ki'])
        
        # Primary reaction: r1 = k_eff * Ca [mol/(L·min)]
        r1 = k_effective * Ca
        
        # Additional nonlinear terms
        if p.get('nonlinear_mode', False):
            # Autocatalytic/second-order reaction
            r2 = p['k2'] * Ca**2
            r_total = r1 + r2
            Q_rxn = r1 * (-p['dH']) / (p['rho'] * p['Cp']) + r2 * (-p['dH2']) / (p['rho'] * p['Cp'])
        else:
            r_total = r1
            Q_rxn = r1 * (-p['dH']) / (p['rho'] * p['Cp'])
        
        # Material Balance: dCa/dt = inflow - outflow - consumption
        dCadt = (p['q']/p['V']) * (p['Cai'] - Ca) - r_total
        
        # Energy Balance: dT/dt = flow + reaction heat + cooling
        dTdt = (p['q']/p['V']) * (p['Ti'] - T) + Q_rxn + (p['UA'] / (p['V'] * p['rho'] * p['Cp'])) * (u_Tc - T)
        
        return [dCadt, dTdt]
    
    def jacobian(self, x, u_Tc, dt):
        """
        Compute linearized state transition matrix F.
        
        Parameters:
        -----------
        x : array_like, shape (2,)
            State vector [Ca, T]
        u_Tc : float
            Coolant temperature
        dt : float
            Time step for discretization
            
        Returns:
        --------
        F : ndarray, shape (2, 2)
            Discretized Jacobian matrix
        """
        p = self.params
        Ca, T = x
        Ca = max(0.0, Ca)
        T = max(273.0, T)
        
        # Primary rate with inhibition
        k_primary = p['k0'] * np.exp(-p['E'] / (p['R_gas'] * T))
        dk_primary_dT = k_primary * (p['E'] / (p['R_gas'] * T**2))
        
        # Substrate inhibition
        inhibition_factor = 1 / (1 + Ca / p['Ki'])
        k_eff = k_primary * inhibition_factor
        
        # Derivatives of k_eff
        dk_eff_dCa = -k_primary / (p['Ki'] * (1 + Ca/p['Ki'])**2)
        dk_eff_dT = dk_primary_dT * inhibition_factor
        
        # Reaction rate derivatives
        dr1_dCa = k_eff + Ca * dk_eff_dCa
        dr1_dT = Ca * dk_eff_dT
        
        if p.get('nonlinear_mode', False):
            # Second-order reaction derivatives
            dr2_dCa = 2 * p['k2'] * Ca
            dr2_dT = 0.0
            
            dr_total_dCa = dr1_dCa + dr2_dCa
            dr_total_dT = dr1_dT + dr2_dT
            
            # Heat generation derivatives
            alpha1 = (-p['dH']) / (p['rho'] * p['Cp'])
            alpha2 = (-p['dH2']) / (p['rho'] * p['Cp'])
            dQ_rxn_dCa = alpha1 * dr1_dCa + alpha2 * dr2_dCa
            dQ_rxn_dT = alpha1 * dr1_dT + alpha2 * dr2_dT
        else:
            dr_total_dCa = dr1_dCa
            dr_total_dT = dr1_dT
            
            alpha1 = (-p['dH']) / (p['rho'] * p['Cp'])
            dQ_rxn_dCa = alpha1 * dr1_dCa
            dQ_rxn_dT = alpha1 * dr1_dT
        
        # Cooling coefficient
        beta = p['UA'] / (p['V'] * p['rho'] * p['Cp'])
        
        # Continuous Jacobian A = ∂f/∂x
        a11 = -p['q']/p['V'] - dr_total_dCa
        a12 = -dr_total_dT
        a21 = dQ_rxn_dCa
        a22 = -p['q']/p['V'] + dQ_rxn_dT - beta
        
        A = np.array([[a11, a12], [a21, a22]])
        
        # Discretize: F = I + A*dt (first-order Euler)
        F = np.eye(2) + A * dt
        return F
    
    def get_state_names(self):
        """Return names of state variables"""
        return ['Concentration (Ca)', 'Temperature (T)']
    
    def get_state_units(self):
        """Return units of state variables"""
        return ['mol/L', 'K']

