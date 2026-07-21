"""
Extended time-dependent amplitude model with:
1. Time resolution calibration (scale factor + bias)
2. Flavour tagging calibration (multiple taggers, per-year parameters)

This module extends tf-pwa's time_dep.py to match LHCb Run2 Bs->J/psi Phi analysis.
"""

import math
import json
import numpy as np
import tensorflow as tf

from tf_pwa.amp.time_dep import (
    TimeDepParamsConvAmplitudeModel,
    TimeDepCpConvAmplitudeModel,
    conv_exp_gaussian,
    conv_exp_gaussian_complex,
)
from tf_pwa.amp.amp import (
    BaseAmplitudeModel,
    create_amplitude,
    register_amp_model,
)
from tf_pwa.amp.core import HelicityDecay, register_decay
from tf_pwa.config import get_config


# =============================================================================
# 1. Time Resolution Calibration
# =============================================================================

def conv_exp_gaussian_with_resolution(t, sigma, gamma, t_min=0.0,
                                      res_p0=1.0, res_p1=0.0, time_bias=0.0):
    """
    Convolution with Gaussian resolution, including scale factor and time bias.
    
    The effective sigma is: sigma_eff = res_p0 + res_p1 * sigma
    The effective time is: t_eff = t - time_bias
    
    Args:
        t: measured time
        sigma: per-event time error
        gamma: decay width
        t_min: minimum time
        res_p0: resolution scale parameter 0 (default 1.0)
        res_p1: resolution scale parameter 1 (default 0.0)
        time_bias: time bias (default 0.0)
        
    Returns:
        Convolution result
    """
    # Apply time bias
    t_eff = t - time_bias
    
    # Apply resolution scale factor
    sigma_eff = res_p0 + res_p1 * sigma
    
    # Use the base convolution function
    return conv_exp_gaussian(t_eff, sigma_eff, gamma, t_min)


def conv_exp_gaussian_complex_with_resolution(t, sigma, gamma, delta_m, t_min=0.0,
                                              res_p0=1.0, res_p1=0.0, time_bias=0.0):
    """
    Complex convolution with Gaussian resolution, including scale factor and time bias.
    """
    t_eff = t - time_bias
    sigma_eff = res_p0 + res_p1 * sigma
    return conv_exp_gaussian_complex(t_eff, sigma_eff, gamma, delta_m, t_min)


@register_amp_model("time_dep_cp_conv_res")
class TimeDepCpConvResAmplitudeModel(TimeDepCpConvAmplitudeModel):
    """
    Time-dependent CP-violating amplitude with time resolution calibration.
    
    Extends TimeDepCpConvAmplitudeModel with:
    - Time resolution scale factor: sigma_eff = res_p0 + res_p1 * sigma
    - Time bias: t_eff = t - time_bias
    
    Attributes:
        res_p0: Resolution scale parameter 0 (typically ~1.0)
        res_p1: Resolution scale parameter 1 (typically small)
        time_bias: Time measurement bias (typically small)
    """
    
    def __init__(self, *args, use_resolution=True, **kwargs):
        """
        Args:
            use_resolution: If True, apply time resolution calibration
        """
        self.use_resolution = use_resolution
        super().__init__(*args, **kwargs)
    
    def init_params(self, *args, **kwargs):
        super().init_params(*args, **kwargs)
        top = self.decay_group.top
        
        # Time resolution parameters
        if self.use_resolution:
            top.res_p0 = top.add_var("res_p0", value=1.0, fix=False)
            top.res_p1 = top.add_var("res_p1", value=0.0, fix=True)
            top.time_bias = top.add_var("time_bias", value=0.0, fix=True)
        else:
            top.res_p0 = 1.0
            top.res_p1 = 0.0
            top.time_bias = 0.0
    
    def eval_P_Pbar_time(self, data):
        """
        Evaluate P and Pbar with time resolution calibration.
        """
        A, Abar = self.eval_A_Abar(data)
        top = self.decay_group.top
        phase = top.poq()
        
        ones = tf.ones((1,), dtype=get_config("dtype"))
        t = data.get("time", 0.0 * ones)
        sigma = data.get("time_sigma", 0.0 * ones)
        
        # Get resolution parameters
        if self.use_resolution:
            res_p0 = top.res_p0()
            res_p1 = top.res_p1()
            time_bias = top.time_bias()
        else:
            res_p0 = 1.0
            res_p1 = 0.0
            time_bias = 0.0
        
        # Apply time bias
        t_eff = t - time_bias
        
        # Apply resolution scale factor
        sigma_eff = res_p0 + res_p1 * sigma
        
        # Convolution with effective sigma and time
        # -(Gamma - Delta_Gamma/2)
        exp_pgamma_t = conv_exp_gaussian(
            t_eff, sigma_eff, top.gamma() - top.delta_gamma() / 2, self.t_min
        )
        # -(Gamma + Delta_Gamma/2)
        exp_mgamma_t = conv_exp_gaussian(
            t_eff, sigma_eff, top.gamma() + top.delta_gamma() / 2, self.t_min
        )
        # Complex part for Delta m
        exp_dm_t = conv_exp_gaussian_complex(
            t_eff, sigma_eff, top.gamma(), -top.delta_m(), self.t_min
        )
        
        # Rest of the calculation (same as parent class)
        cosht = (exp_pgamma_t + exp_mgamma_t) / 2
        sinht = (exp_pgamma_t - exp_mgamma_t) / 2
        cost = tf.math.real(exp_dm_t)
        sint = tf.math.imag(exp_dm_t)
        
        A2 = self.decay_group.sum_with_polarization(A)
        Abar2 = self.decay_group.sum_with_polarization(phase * Abar)
        Asum = A2 + Abar2
        Asub = A2 - Abar2
        ReA = self.decay_group.sum_with_polarization(phase * Abar, A)
        ImA = self.decay_group.sum_with_polarization(
            phase * Abar, 1j * A
        )
        
        ret1 = (
            Asum * cosht + Asub * cost - 2 * ReA * sinht - 2 * ImA * sint
        ) / 2
        
        # A <-> Abar, q/p <-> p/q
        A2_p = self.decay_group.sum_with_polarization(Abar)
        Abar2_p = self.decay_group.sum_with_polarization(A / phase)
        Asum_p = A2_p + Abar2_p
        Asub_p = A2_p - Abar2_p
        ReA_p = self.decay_group.sum_with_polarization(A / phase, Abar)
        ImA_p = self.decay_group.sum_with_polarization(A / phase, 1j * Abar)
        
        ret2 = (
            Asum_p * cosht
            + Asub_p * cost
            - 2 * ReA_p * sinht
            - 2 * ImA_p * sint
        ) / 2
        
        # Apply acceptance (if set)
        if hasattr(self, 'time_acceptance') and self.time_acceptance is not None:
            trigger = data.get("trigger", tf.zeros_like(t))
            eps_t = self.time_acceptance(t, trigger=trigger)
            ret1 = ret1 * eps_t
            ret2 = ret2 * eps_t
            
        if hasattr(self, 'angular_acceptance') and self.angular_acceptance is not None:
            angles = data.get("p4", None)
            if angles is not None:
                theta_L = angles[..., 0]
                theta_K = angles[..., 1]
                phi = angles[..., 2]
                ctL = tf.cos(theta_L)
                ctK = tf.cos(theta_K)
                trigger = data.get("trigger", tf.zeros_like(t))
                eps_omega = self.angular_acceptance(ctK, ctL, phi, trigger=trigger)
                ret1 = ret1 * eps_omega
                ret2 = ret2 * eps_omega
        
        return ret1, ret2


# =============================================================================
# 2. Flavour Tagging Calibration (LHCb Run2 style)
# =============================================================================

@register_amp_model("flavour_tag_lhcb")
class FlavourTagLHCbPDF(BaseAmplitudeModel):
    """
    Flavour tagging calibration PDF following LHCb Run2 Bs->J/psi Phi style.
    
    Calibration formula (following RooDalitzTimeCPCBTAG.cxx):
        When tag = +1 (Bs):  eta_cal = (p0 + dp0/2) + (p1 + dp1/2) * (eta - eta_mean)
        When tag = -1 (Bsbar): eta_cal = (p0 - dp0/2) + (p1 - dp1/2) * (eta - eta_mean)
    
    The tagging PDF is computed as:
        P(tag_dec | tag_true) = (1 - eta) if tag_dec == tag_true
                                eta      if tag_dec != tag_true
                                0.5      if tag_dec == 0 (untagged)
    
    For combined taggers (OS + SS):
        Following C++ CombineTag() logic:
        - If only one tagger fires: use that tagger's calibration
        - If both taggers fire with same decision: combine using product rule
        - If both taggers fire with different decisions: choose lower mistag
    
    Attributes:
        taggers: List of tagger configurations, each with:
            - name: Tagger name (e.g., "OS", "SSK", "IFT")
            - eta_name: Variable name for raw eta
            - eta_mean: Mean eta value
            - p0, p1: Base calibration parameters
            - dp0, dp1: Delta parameters (asymmetry corrections)
            - year: Data taking year (2015, 2016, 2017, 2018)
        tag_eff: Tagging efficiency (epsilon_tag)
        prod_asym: Production asymmetry (A_prod)
    """
    
    def __init__(self, *args, tagger_configs=None, tag_eff=1.0, prod_asym=0.0, **kwargs):
        """
        Args:
            tagger_configs: List of dicts, each containing:
                {
                    "name": "OS",
                    "eta_name": "eta_OS",
                    "eta_mean": 0.3546,
                    "p0": 0.3831, "p1": 0.8518,   # Base parameters
                    "dp0": 0.0092, "dp1": 0.0141,  # Delta parameters
                    "year": 2016
                }
            tag_eff: Tagging efficiency (0 <= tag_eff <= 1)
            prod_asym: Production asymmetry A_prod
        """
        self.tagger_configs = tagger_configs or []
        self.tag_eff = tag_eff
        self.prod_asym = prod_asym
        super().__init__(*args, **kwargs)
    
    def init_params(self, *args, **kwargs):
        super().init_params(*args, **kwargs)
        top = self.decay_group.top
        
        # Create calibration parameters for each tagger
        for i, config in enumerate(self.tagger_configs):
            name = config["name"]
            year = config.get("year", 2016)
            
            # Base parameters (p0, p1)
            setattr(top, f"p0_{name}_{year}", 
                    top.add_var(f"p0_{name}_{year}", value=config.get("p0", 0.38)))
            setattr(top, f"p1_{name}_{year}", 
                    top.add_var(f"p1_{name}_{year}", value=config.get("p1", 0.85)))
            
            # Delta parameters (dp0, dp1 for asymmetry)
            setattr(top, f"dp0_{name}_{year}", 
                    top.add_var(f"dp0_{name}_{year}", value=config.get("dp0", 0.0), fix=True))
            setattr(top, f"dp1_{name}_{year}", 
                    top.add_var(f"dp1_{name}_{year}", value=config.get("dp1", 0.0), fix=True))
    
    def calibrate_eta(self, eta, eta_mean, p0, p1, dp0=0.0, dp1=0.0, tag=1):
        """
        Apply calibration with tag-dependent parameters.
        
        Following RooDalitzTimeCPCBTAG.cxx logic:
            When tag = +1 (Bs):   p0_eff = p0 + dp0/2,  p1_eff = p1 + dp1/2
            When tag = -1 (Bsbar): p0_eff = p0 - dp0/2,  p1_eff = p1 - dp1/2
        
        Args:
            eta: Raw mistag probability
            eta_mean: Mean eta value for the tagger
            p0, p1: Base calibration parameters
            dp0, dp1: Delta parameters (asymmetry corrections)
            tag: True tag value (+1 for Bs, -1 for Bsbar)
            
        Returns:
            Calibrated mistag probability
        """
        # Select tag-dependent effective parameters
        p0_eff = tf.where(tag > 0, p0 + dp0 / 2.0, p0 - dp0 / 2.0)
        p1_eff = tf.where(tag > 0, p1 + dp1 / 2.0, p1 - dp1 / 2.0)
        
        return p0_eff + p1_eff * (eta - eta_mean)
    
    def combine_taggers(self, tag_dec_list, eta_cal_list):
        """
        Combine multiple taggers following C++ CombineTag() logic.
        
        Implementation matching RooDalitzTimeCPCBTAG::CombineTag():
            - If both taggers return 0: tag=0, eta=0.5
            - If only tagger1 fires: tag=tag1, eta=eta1
            - If only tagger2 fires: tag=tag2, eta=eta2
            - If both fire and tag1==tag2:
                cbwp = wp1*wp2/(wp1*wp2+(1-wp1)*(1-wp2))
                cbwm = wm1*wm2/(wm1*wm2+(1-wm1)*(1-wm2))
                eta_comb = (cbwp + cbwm)/2
                delta_eta = cbwp - cbwm
            - If both fire and tag1!=tag2:
                Choose tagger with lower mistag
                cbwp = wp1*(1-wp2)/(wp1*(1-wp2)+(1-wp1)*wp2)
                cbwm = wm1*(1-wm2)/(wm1*(1-wm2)+(1-wm1)*wm2)
        
        Args:
            tag_dec_list: List of tag decisions (tag1, tag2)
            eta_cal_list: List of calibrated eta values (eta1, eta2)
            
        Returns:
            Combined tag decision and calibrated eta
        """
        if len(tag_dec_list) == 1:
            return tag_dec_list[0], eta_cal_list[0]
        
        tag1, tag2 = tag_dec_list
        eta1, eta2 = eta_cal_list
        
        # wp = eta + delta_eta/2 for correct tag
        # wm = eta - delta_eta/2 for wrong tag
        # For single tagger, delta_eta is typically small, so wp ~ wm ~ eta
        
        # Case 1: Both untagged
        mask_both_zero = tf.logical_and(tag1 == 0, tag2 == 0)
        comb_tag_zero = tf.constant(0, dtype=tag1.dtype)
        comb_eta_zero = tf.constant(0.5, dtype=eta1.dtype)
        
        # Case 2: Only tagger1 fires
        mask_only1 = tf.logical_and(tag1 != 0, tag2 == 0)
        comb_tag_only1 = tag1
        comb_eta_only1 = eta1
        
        # Case 3: Only tagger2 fires
        mask_only2 = tf.logical_and(tag1 == 0, tag2 != 0)
        comb_tag_only2 = tag2
        comb_eta_only2 = eta2
        
        # Case 4: Both fire
        mask_both_fire = tf.logical_and(tag1 != 0, tag2 != 0)
        
        # Subcase 4a: Same tag decision
        mask_same_tag = tf.logical_and(mask_both_fire, tag1 == tag2)
        
        wp1 = eta1 + 0.0  # delta_eta approx 0 for single tagger
        wm1 = eta1 - 0.0
        wp2 = eta2 + 0.0
        wm2 = eta2 - 0.0
        
        denom_same_p = wp1 * wp2 + (1 - wp1) * (1 - wp2)
        denom_same_m = wm1 * wm2 + (1 - wm1) * (1 - wm2)
        
        cbwp_same = wp1 * wp2 / denom_same_p
        cbwm_same = wm1 * wm2 / denom_same_m
        comb_eta_same = (cbwp_same + cbwm_same) / 2.0
        comb_tag_same = tag1
        
        # Subcase 4b: Different tag decisions
        mask_diff_tag = tf.logical_and(mask_both_fire, tag1 != tag2)
        
        # Choose tagger with lower mistag
        use_tagger1 = eta1 < eta2
        chosen_tag = tf.where(use_tagger1, tag1, tag2)
        
        denom_diff_p = tf.where(use_tagger1, 
                                wp1 * (1 - wp2) + (1 - wp1) * wp2,
                                wp2 * (1 - wp1) + (1 - wp2) * wp1)
        denom_diff_m = tf.where(use_tagger1,
                                wm1 * (1 - wm2) + (1 - wm1) * wm2,
                                wm2 * (1 - wm1) + (1 - wm2) * wm1)
        
        cbwp_diff = tf.where(use_tagger1,
                             wp1 * (1 - wp2) / denom_diff_p,
                             wp2 * (1 - wp1) / denom_diff_p)
        cbwm_diff = tf.where(use_tagger1,
                             wm1 * (1 - wm2) / denom_diff_m,
                             wm2 * (1 - wm1) / denom_diff_m)
        comb_eta_diff = (cbwp_diff + cbwm_diff) / 2.0
        comb_tag_diff = chosen_tag
        
        # Combine all cases
        comb_tag = tf.where(mask_both_zero, comb_tag_zero,
                           tf.where(mask_only1, comb_tag_only1,
                                   tf.where(mask_only2, comb_tag_only2,
                                           tf.where(mask_same_tag, comb_tag_same, comb_tag_diff))))
        
        comb_eta = tf.where(mask_both_zero, comb_eta_zero,
                           tf.where(mask_only1, comb_eta_only1,
                                   tf.where(mask_only2, comb_eta_only2,
                                           tf.where(mask_same_tag, comb_eta_same, comb_eta_diff))))
        
        return comb_tag, comb_eta
    
    def pdf(self, data):
        """
        Compute flavour tagging PDF including tagging efficiency and production asymmetry.
        
        Full decay rate formula (following LHCb conventions):
            dGamma/dt = (1 ± A_prod) * [ (1 - eta) * P(t) + eta * Pbar(t) ] for tag=±1
            dGamma/dt = (P(t) + Pbar(t)) / 2 for tag=0
        
        The tagging PDF factor is:
            P(tag_dec | tag_true) = tag_eff * P_correct + (1 - tag_eff) * P_random
        
        Args:
            data: Dictionary containing:
                - tag: True tag value (+1, -1)
                - tag_value: Tag decision from tagger (+1, -1, 0)
                - eta_*: Raw mistag probabilities for each tagger
                - trigger: Optional trigger flag
                
        Returns:
            Flavour tagging PDF factor
        """
        if not self.tagger_configs:
            ones = tf.ones_like(data.get("time", 1.0))
            return ones
        
        # Get true tag
        true_tag = data.get("tag", 1.0)
        
        # Process each tagger
        tag_dec_list = []
        eta_cal_list = []
        
        for i, config in enumerate(self.tagger_configs):
            name = config["name"]
            year = config.get("year", 2016)
            eta_name = config.get("eta_name", f"eta_{name}")
            tag_name = config.get("tag_name", f"tag_{name}")
            eta_mean = config.get("eta_mean", 0.35)
            
            # Get raw eta and tag decision
            eta = data.get(eta_name, eta_mean)
            tag_dec = data.get(tag_name, 0.0)
            
            # Get calibration parameters
            top = self.decay_group.top
            p0 = getattr(top, f"p0_{name}_{year}")()
            p1 = getattr(top, f"p1_{name}_{year}")()
            dp0 = getattr(top, f"dp0_{name}_{year}")()
            dp1 = getattr(top, f"dp1_{name}_{year}")()
            
            # Calibrate eta with tag-dependent parameters
            eta_cal = self.calibrate_eta(eta, eta_mean, p0, p1, dp0, dp1, true_tag)
            
            # Clip to [0, 0.5]
            eta_cal = tf.clip_by_value(eta_cal, 0.0, 0.5)
            
            tag_dec_list.append(tag_dec)
            eta_cal_list.append(eta_cal)
        
        # Combine taggers
        comb_tag, comb_eta = self.combine_taggers(tag_dec_list, eta_cal_list)
        
        # Compute tagging PDF
        # P(tag_dec | tag_true) = 1 - eta if correct, eta if wrong, 0.5 if untagged
        same_tag = comb_tag * true_tag > 0
        is_untagged = comb_tag == 0
        
        prob = tf.where(
            is_untagged,
            0.5,  # Untagged
            tf.where(
                same_tag,
                1 - comb_eta,  # Correct tag
                comb_eta  # Wrong tag
            )
        )
        
        # Apply tagging efficiency
        # P_eff = tag_eff * P_calibrated + (1 - tag_eff) * P_random
        # P_random = 0.5 for all cases (random guess)
        prob_eff = self.tag_eff * prob + (1 - self.tag_eff) * 0.5
        
        return prob_eff


# =============================================================================
# 3. Combined Model: Time Resolution + Flavour Tagging + Time-Dependent Amplitude
# =============================================================================

@register_amp_model("time_dep_cp_conv_res_tag")
class TimeDepCpConvResTagAmplitudeModel(TimeDepCpConvResAmplitudeModel):
    """
    Complete time-dependent CP-violating amplitude model with:
    - Time resolution calibration
    - Flavour tagging calibration
    - Time and angular acceptance
    
    This is the full model for LHCb Run2 Bs->J/psi Phi analysis.
    """
    
    def __init__(self, *args, tagger_configs=None, tag_eff=1.0, **kwargs):
        self.tagger_configs = tagger_configs or []
        self.tag_eff = tag_eff
        super().__init__(*args, **kwargs)
    
    def init_params(self, *args, **kwargs):
        super().init_params(*args, **kwargs)
        
        # Add flavour tagging calibration parameters
        top = self.decay_group.top
        for i, config in enumerate(self.tagger_configs):
            name = config["name"]
            year = config.get("year", 2016)
            
            if not hasattr(top, f"p0_{name}_{year}"):
                setattr(top, f"p0_{name}_{year}", 
                        top.add_var(f"p0_{name}_{year}", value=config.get("p0", 0.38)))
                setattr(top, f"p1_{name}_{year}", 
                        top.add_var(f"p1_{name}_{year}", value=config.get("p1", 0.85)))
                setattr(top, f"p0bar_{name}_{year}", 
                        top.add_var(f"p0bar_{name}_{year}", value=config.get("p0bar", config.get("p0", 0.38))))
                setattr(top, f"p1bar_{name}_{year}", 
                        top.add_var(f"p1bar_{name}_{year}", value=config.get("p1bar", config.get("p1", 0.85))))
    
    def pdf(self, data):
        """
        Combined PDF: time-dependent amplitude * flavour tagging PDF
        """
        # Get time-dependent amplitude
        P, Pbar = self.eval_P_Pbar_time(data)
        
        # Normalization (using MC weights)
        # This is handled by the base class
        
        # Get flavour tagging PDF
        if self.tagger_configs:
            tag_pdf = FlavourTagLHCbPDF(
                self.decay_group,
                tagger_configs=self.tagger_configs,
                tag_eff=self.tag_eff
            )
            tag_factor = tag_pdf.pdf(data)
        else:
            ones = tf.ones_like(data.get("time", 1.0))
            tag_factor = ones
        
        # Combine
        return (P + Pbar) * tag_factor


# =============================================================================
# 4. Utility Functions for Loading Calibration Parameters
# =============================================================================

def load_time_resolution_params(json_path):
    """
    Load time resolution parameters from JSON file.
    
    Expected format:
        {
            "TimeResParameters": [
                {"Name": "p0", "Value": 1.0, "Error": 0.01},
                {"Name": "p1", "Value": 0.0, "Error": 0.01},
                {"Name": "rho_p0_p1_time_res", "Value": 0.0, "Error": 0.0}
            ],
            "TimeBias": {
                "name": "meanshift",
                "Value": 0.0,
                "Error": 0.001
            }
        }
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        dict with keys: res_p0, res_p1, time_bias, res_corr (correlation)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    params = {}
    
    # Time resolution parameters
    if "TimeResParameters" in data:
        for param in data["TimeResParameters"]:
            if param["Name"] == "p0":
                params["res_p0"] = param["Value"]
                params["res_p0_err"] = param["Error"]
            elif param["Name"] == "p1":
                params["res_p1"] = param["Value"]
                params["res_p1_err"] = param["Error"]
            elif param["Name"] == "rho_p0_p1_time_res":
                params["res_corr"] = param["Value"]
    
    # Time bias
    if "TimeBias" in data:
        params["time_bias"] = data["TimeBias"]["Value"]
        params["time_bias_err"] = data["TimeBias"]["Error"]
    
    return params


def load_tagging_params(json_path):
    """
    Load tagging calibration parameters from JSON file.
    
    Expected format (LHCb Run2 style):
        {
            "TaggingParameters": {
                "TaggingOS": {
                    "Parameter_ETA": {"Name": "eta_OS_Run2", "Value": 0.3546},
                    "Parameter": [
                        {"Name": "p0_OS_2016", "Value": 0.3831, "Error": 0.0007},
                        {"Name": "p1_OS_2016", "Value": 0.8518, "Error": 0.0062},
                        ...
                    ],
                    "StatisticalCorrelationMatrix": [...]
                },
                "TaggingSSK": {...},
                "TaggingIFT": {...}
            }
        }
    
    Args:
        json_path: Path to JSON file (fit_inputs_YYYY.json)
        
    Returns:
        tagger_configs: List of tagger configuration dicts
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    tagger_configs = []
    
    if "TaggingParameters" not in data:
        return tagger_configs
    
    tagging_params = data["TaggingParameters"]
    
    # Process each tagger
    for tagger_name in ["TaggingOS", "TaggingSSK", "TaggingIFT"]:
        if tagger_name not in tagging_params:
            continue
        
        tagger_data = tagging_params[tagger_name]
        
        # Get eta_mean
        eta_mean = tagger_data["Parameter_ETA"]["Value"]
        
        # Get parameters (base and deltas for each year)
        params_list = tagger_data["Parameter"]
        
        # Organize by year
        year_params = {}
        for param in params_list:
            name = param["Name"]
            # Parse: p0_OS_2016, p1_OS_2016, dp0_OS_2016, dp1_OS_2016
            parts = name.split("_")
            if len(parts) >= 3:
                year = int(parts[-1])
                param_type = "_".join(parts[:-1])  # p0_OS, p1_OS, dp0_OS, dp1_OS
                
                if year not in year_params:
                    year_params[year] = {}
                
                year_params[year][param_type] = param["Value"]
        
        # Create config for each year
        for year, params in year_params.items():
            config = {
                "name": tagger_name.replace("Tagging", ""),
                "eta_name": f"eta_{tagger_name.replace('Tagging', '')}",
                "eta_mean": eta_mean,
                "year": year,
                "p0": params.get(f"p0_{tagger_name.replace('Tagging', '')}", 0.38),
                "p1": params.get(f"p1_{tagger_name.replace('Tagging', '')}", 0.85),
                "dp0": params.get(f"dp0_{tagger_name.replace('Tagging', '')}", 0.0),
                "dp1": params.get(f"dp1_{tagger_name.replace('Tagging', '')}", 0.0),
            }
            
            # Adjust p0, p1 with dp0, dp1 if present
            base_p0 = config["p0"]
            base_p1 = config["p1"]
            config["p0"] = base_p0 + config["dp0"]
            config["p1"] = base_p1 + config["dp1"]
            
            tagger_configs.append(config)
    
    return tagger_configs


def create_tagger_configs_from_json(json_path, taggers=["OS", "SSK"], years=[2016, 2017, 2018]):
    """
    Create tagger configuration list from LHCb Run2 JSON format.
    
    Args:
        json_path: Path to fit_inputs JSON file
        taggers: List of tagger names to include ("OS", "SSK", "IFT")
        years: List of years to include
        
    Returns:
        List of tagger configuration dicts
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    configs = []
    
    if "TaggingParameters" not in data:
        print(f"Warning: No TaggingParameters found in {json_path}")
        return configs
    
    for tagger in taggers:
        tagger_key = f"Tagging{tagger}"
        if tagger_key not in data["TaggingParameters"]:
            continue
        
        tagger_data = data["TaggingParameters"][tagger_key]
        eta_mean = tagger_data["Parameter_ETA"]["Value"]
        
        # Get base parameters (from first year or average)
        base_params = {}
        for param in tagger_data["Parameter"]:
            name = param["Name"]
            # Find base parameters (without year suffix or first year)
            if f"_{tagger}_" in name:
                parts = name.split(f"_{tagger}_")
                if len(parts) == 2:
                    year_str = parts[1]
                    if year_str.isdigit():
                        year = int(year_str)
                        param_type = name.split(f"_{year}")[0].split("_")[-1]
                        
                        if year not in base_params:
                            base_params[year] = {}
                        base_params[year][param_type] = param["Value"]
        
        # Create configs for each year
        for year in years:
            if year not in base_params:
                continue
            
            params = base_params[year]
            
            config = {
                "name": tagger,
                "eta_name": f"eta_{tagger}",
                "eta_mean": eta_mean,
                "year": year,
                "p0": params.get("p0", 0.38),
                "p1": params.get("p1", 0.85),
                "p0bar": params.get("p0bar", params.get("p0", 0.38)),
                "p1bar": params.get("p1bar", params.get("p1", 0.85)),
            }
            
            configs.append(config)
    
    return configs


# =============================================================================
# 5. Example Usage and Test Functions
# =============================================================================

def test_time_resolution():
    """
    Test time resolution calibration.
    """
    import numpy as np
    
    # Create test data
    t = np.random.exponential(1.0/0.65789, 1000)  # Bs lifetime ~1.5 ps
    sigma = np.random.normal(0.04, 0.01, 1000)  # Typical per-event sigma
    
    # Convert to tensors
    t_tf = tf.constant(t, dtype=tf.float32)
    sigma_tf = tf.constant(sigma, dtype=tf.float32)
    
    # Test convolution with resolution
    result = conv_exp_gaussian_with_resolution(
        t_tf, sigma_tf, gamma=0.65789, t_min=0.3,
        res_p0=1.0, res_p1=0.0, time_bias=0.0
    )
    
    print(f"Convolution result shape: {result.shape}")
    print(f"Mean result: {tf.reduce_mean(result)}")
    
    return result


def test_tagging_calibration():
    """
    Test flavour tagging calibration.
    """
    import numpy as np
    
    # Create test data
    eta = np.random.uniform(0.0, 0.5, 1000)
    true_tag = np.random.choice([-1, 1], 1000)
    
    # Calibrate
    eta_mean = 0.3546
    p0 = 0.3831
    p1 = 0.8518
    
    eta_cal = p0 + p1 * (eta - eta_mean)
    eta_cal = np.clip(eta_cal, 0.0, 0.5)
    
    print(f"Raw eta: mean={np.mean(eta):.4f}")
    print(f"Calibrated eta: mean={np.mean(eta_cal):.4f}")
    
    return eta_cal


if __name__ == "__main__":
    print("Testing time resolution calibration...")
    test_time_resolution()
    
    print("\nTesting flavour tagging calibration...")
    test_tagging_calibration()
    
    print("\nDone!")