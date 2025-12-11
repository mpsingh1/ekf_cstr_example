"""
Coolant temperature profile generators.

Provides various disturbance scenarios for testing EKF performance.
"""

import numpy as np


def generate_coolant_profile(profile_type, t_array, params, dt):
    """
    Generate coolant temperature profile based on selected type.
    
    Parameters:
    -----------
    profile_type : str
        Type of profile ('Step Change', 'Sinusoidal', 'Random Walk', etc.)
    t_array : ndarray
        Time array
    params : dict
        Profile parameters
    dt : float
        Time step
        
    Returns:
    --------
    Tc_profile : ndarray
        Coolant temperature at each time step
    """
    if profile_type == "Step Change":
        tc_initial = params.get('tc_initial', 300.0)
        tc_final = params.get('tc_final', 350.0)
        tc_step_time = params.get('tc_step_time', 5.0)
        return np.where(t_array < tc_step_time, tc_initial, tc_final)
    
    elif profile_type == "Sinusoidal":
        tc_mean = params.get('tc_mean', 325.0)
        tc_amplitude = params.get('tc_amplitude', 25.0)
        tc_period = params.get('tc_period', 8.0)
        omega = 2 * np.pi / tc_period
        return tc_mean + tc_amplitude * np.sin(omega * t_array)
    
    elif profile_type == "Random Walk":
        tc_start = params.get('tc_start', 300.0)
        tc_volatility = params.get('tc_volatility', 1.0)
        tc_min = params.get('tc_min', 290.0)
        tc_max = params.get('tc_max', 360.0)
        
        Tc_profile = np.zeros(len(t_array))
        Tc_profile[0] = tc_start
        for i in range(1, len(t_array)):
            Tc_profile[i] = Tc_profile[i-1] + np.random.normal(0, tc_volatility * dt)
            Tc_profile[i] = np.clip(Tc_profile[i], tc_min, tc_max)
        return Tc_profile
    
    elif profile_type == "Multiple Steps":
        tc_base = params.get('tc_base', 300.0)
        tc_step_size = params.get('tc_step_size', 20.0)
        tc_step_interval = params.get('tc_step_interval', 3.0)
        
        Tc_profile = np.full(len(t_array), tc_base)
        step_count = 0
        for i, t in enumerate(t_array):
            if t > 0 and t % tc_step_interval < dt:
                step_count += 1
            Tc_profile[i] = tc_base + step_count * tc_step_size * (1 if step_count % 2 == 1 else -1)
        return Tc_profile
    
    elif profile_type == "Ramp":
        tc_start = params.get('tc_start', 300.0)
        tc_end = params.get('tc_end', 360.0)
        sim_time = params.get('sim_time', t_array[-1])
        return tc_start + (tc_end - tc_start) * t_array / sim_time
    
    elif profile_type == "Constant":
        tc_constant = params.get('tc_constant', 320.0)
        return np.full(len(t_array), tc_constant)
    
    # Default fallback
    return np.full(len(t_array), 300.0)

