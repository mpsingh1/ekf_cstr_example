"""
Extended Kalman Filter Implementation

References:
- Welch, G., & Bishop, G. (2006). An Introduction to the Kalman Filter.
- Simon, D. (2006). Optimal State Estimation.
"""

import numpy as np
from scipy.integrate import solve_ivp


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear systems.
    
    Uses linearization (Jacobian) for covariance propagation while
    maintaining full nonlinear model for state prediction.
    """
    
    def __init__(self, model, H, Q, R, P0, x0):
        """
        Initialize Extended Kalman Filter.
        
        Parameters:
        -----------
        model : object
            Model object with methods: dynamics(t, x, u) and jacobian(x, u, dt)
        H : ndarray
            Measurement matrix
        Q : ndarray
            Process noise covariance
        R : ndarray
            Measurement noise covariance
        P0 : ndarray
            Initial state covariance
        x0 : ndarray
            Initial state estimate
        """
        self.model = model
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P0.copy()
        self.x = x0.copy()
        
        # For tracking
        self.innovation = None
        self.kalman_gain = None
        
    def predict(self, u, dt):
        """
        Prediction step: propagate state and covariance forward.
        
        Parameters:
        -----------
        u : float or array_like
            Control input
        dt : float
            Time step
            
        Returns:
        --------
        x_pred : ndarray
            Predicted state
        P_pred : ndarray
            Predicted covariance
        """
        # State prediction using full nonlinear model
        sol = solve_ivp(
            lambda t, x: self.model.dynamics(t, x, u),
            [0, dt],
            self.x,
            method='RK45',
            rtol=1e-6,
            atol=1e-8
        )
        x_pred = sol.y[:, -1]
        
        # Apply physical constraints if model has them
        x_pred = self._apply_constraints(x_pred)
        
        # Covariance prediction using linearized model
        F = self.model.jacobian(self.x, u, dt)
        P_pred = F @ self.P @ F.T + self.Q
        
        return x_pred, P_pred
    
    def update(self, x_pred, P_pred, y_meas):
        """
        Update step: incorporate measurement.
        
        Parameters:
        -----------
        x_pred : ndarray
            Predicted state
        P_pred : ndarray
            Predicted covariance
        y_meas : ndarray
            Measurement vector
            
        Returns:
        --------
        x_est : ndarray
            Updated state estimate
        P_est : ndarray
            Updated covariance
        K : ndarray
            Kalman gain
        """
        # Innovation (measurement residual)
        y_pred = self.H @ x_pred
        residual = y_meas - y_pred
        
        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.R
        
        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # State update
        x_est = x_pred + K @ residual
        
        # Apply physical constraints
        x_est = self._apply_constraints(x_est)
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(len(x_est)) - K @ self.H
        P_est = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        
        # Store for analysis
        self.innovation = residual
        self.kalman_gain = K
        
        return x_est, P_est, K
    
    def step(self, y_meas, u, dt):
        """
        Complete EKF cycle: predict + update.
        
        Parameters:
        -----------
        y_meas : ndarray
            Measurement
        u : float or array_like
            Control input
        dt : float
            Time step
            
        Returns:
        --------
        x_est : ndarray
            Updated state estimate
        """
        # Predict
        x_pred, P_pred = self.predict(u, dt)
        
        # Update
        x_est, P_est, K = self.update(x_pred, P_pred, y_meas)
        
        # Store
        self.x = x_est
        self.P = P_est
        
        return x_est

    def update_only(self, y_meas):
        """
        Measurement update without time propagation.

        Useful at k=0 when you have an initial measurement y_0 but no prior
        control interval to propagate through.
        """
        x_est, P_est, K = self.update(self.x, self.P, y_meas)
        self.x = x_est
        self.P = P_est
        self.kalman_gain = K
        return x_est
    
    def _apply_constraints(self, x):
        """Apply physical constraints to state vector"""
        # CSTR-specific: non-negative concentration, temperature > 0K
        x[0] = max(0.0, x[0])  # Ca >= 0
        x[1] = max(273.0, x[1])  # T >= 273K
        return x
    
    def get_state(self):
        """Get current state estimate"""
        return self.x.copy()
    
    def get_covariance(self):
        """Get current covariance matrix"""
        return self.P.copy()
    
    def get_innovation(self):
        """Get last innovation (measurement residual)"""
        return self.innovation
    
    def get_kalman_gain(self):
        """Get last Kalman gain"""
        return self.kalman_gain

