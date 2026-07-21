"""Time acceptance using cubic B-spline parameterization (LHCb Run2 style)

Supports both Biased and Unbiased trigger categories with per-event selection.

Extrapolation beyond max knot (typically 9.0 ps) uses linear extrapolation
based on the derivative at the boundary, following the method in generate_phsp_fast.py.
"""
import numpy as np
import tensorflow as tf
from tf_pwa.tensorflow_wrapper import tf as tfw
from scipy.interpolate import BSpline


class BSplineTimeAcceptance:
    """
    Cubic B-spline time acceptance: ε(t) = Σ c_i * B_i(t; knots)
    
    Default knots from LHCb Run2: [0.3, 0.91, 1.96, 9.0] ps
    Default coefficients from v4r1/fit_inputs_2015.json
    
    For t > max_knot (typically 9.0 ps), linear extrapolation is applied:
        ε(t) = ε(t_max) + dε/dt(t_max) * (t - t_max)
    where dε/dt is computed using numerical differentiation with step h=1e-3.
    """
    
    def __init__(self, knots=None, coefficients_unbiased=None, coefficients_biased=None, t_min=0.3, t_max=15.0):
        if knots is None:
            self.knots = [0.3, 0.91, 1.96, 9.0]  # LHCb Run2 knots (ps)
        else:
            self.knots = list(knots)

        n_coeffs = len(self.knots) + 2  # 4 knots -> 6 coefficients
        
        # Unbiased coefficients (default: flat acceptance)
        if coefficients_unbiased is None:
            self.coeff_unbiased = [1.0] * n_coeffs
        else:
            self.coeff_unbiased = list(coefficients_unbiased)
        self.coeff_unbiased[0] = 1.0  # coeff0 fixed
        
        # Biased coefficients (default: flat acceptance)
        if coefficients_biased is None:
            self.coeff_biased = [1.0] * n_coeffs
        else:
            self.coeff_biased = list(coefficients_biased)
        self.coeff_biased[0] = 1.0  # coeff0 fixed
        
        self.t_min = t_min
        self.t_max = t_max
        self.max_knot = max(self.knots)  # typically 9.0 ps
        
        # Precompute clamped knot vector for scipy BSpline
        k = 3  # cubic
        self._spline_t = np.array(
            [self.knots[0]] * (k + 1)
            + self.knots[1:-1]
            + [self.knots[-1]] * (k + 1)
        )
        
        # Step size for numerical differentiation in extrapolation
        self._h = 1e-3
    
    def _eval_spline(self, t, coefficients):
        """Evaluate B-spline for a single coefficient set."""
        c = np.array(coefficients, dtype=np.float64)
        spl = BSpline(self._spline_t, c, 3, extrapolate=False)
        result = spl(t)
        return result
    
    def _linear_extrapolate(self, t_np, coefficients, trigger_val=None):
        """
        Apply linear extrapolation for t > max_knot.
        
        Uses numerical differentiation to compute derivative at max_knot:
            ds/dt = (s(t_max) - s(t_max - h)) / h
        
        Args:
            t_np: numpy array of times
            coefficients: spline coefficients
            trigger_val: trigger value (0 or 1) for debug purposes
        
        Returns:
            Array with extrapolation applied where needed
        """
        extrap_mask = t_np > self.max_knot
        
        if not np.any(extrap_mask):
            return self._eval_spline(t_np, coefficients)
        
        # Evaluate spline at max_knot and max_knot - h
        s_at_max = self._eval_spline(np.array([self.max_knot]), coefficients)[0]
        s_at_minus = self._eval_spline(np.array([self.max_knot - self._h]), coefficients)[0]
        
        # Numerical derivative
        ds_dt = (s_at_max - s_at_minus) / self._h
        
        # Evaluate spline for all points first
        result = self._eval_spline(t_np, coefficients)
        
        # Apply linear extrapolation
        extrap_result = s_at_max + ds_dt * (t_np[extrap_mask] - self.max_knot)
        result[extrap_mask] = extrap_result
        
        # Handle NaN from spline extrapolation=False for t < t_min
        result = np.where(np.isnan(result), 0.0, result)
        
        # Clip to non-negative (acceptance cannot be negative)
        result = np.clip(result, 0.0, None)
        
        return result
    
    def __call__(self, t, trigger=None):
        """
        Evaluate ε(t) for given time array, with linear extrapolation for t > 9.0 ps.
        
        Args:
            t: TensorFlow tensor of decay times (ps)
            trigger: Optional tensor of trigger flags (0=unbiased, 1=biased),
                     shape (N,). If None, uses unbiased coefficients.
        Returns:
            TensorFlow tensor of acceptance values
        """
        t_np = t.numpy() if hasattr(t, "numpy") else np.array(t)
        trigger_np = (
            trigger.numpy()
            if trigger is not None and hasattr(trigger, "numpy")
            else None
        )

        # Evaluate splines with extrapolation
        eps_unbiased = self._linear_extrapolate(t_np, self.coeff_unbiased)
        eps_biased = self._linear_extrapolate(t_np, self.coeff_biased)

        # Select per-event based on trigger
        if trigger_np is not None:
            result = np.where(trigger_np == 1, eps_biased, eps_unbiased)
        else:
            result = eps_unbiased

        return tf.constant(result, dtype=t.dtype)