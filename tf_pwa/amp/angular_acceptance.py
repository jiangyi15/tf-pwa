#!/usr/bin/env python3
"""Angular acceptance using 10 transversity-basis weights (LHCb Run2 style).

Supports both Biased and Unbiased trigger categories with per-event selection.

The angular acceptance is parameterized as:
    ε(Ω) = Σ_{k=1}^{10} c_k * f_k(Ω)

where f_k(Ω) are the angular basis functions from LHCb-ANA-2019-032 Table 1.

This implementation follows the Solve-8 formula:
    ε(Ω) = Σ_{k=1}^{10} c_k * f_k(Ω)

where c_k coefficients are computed from measured omega_k values:
    Block A (4×4 inverse matrix): c1, c2, c3, c7
    Blocks B, C, D (diagonal division): c4, c5, c6, c8, c9, c10

See Solve-8 for detailed formulas.
"""
import tensorflow as tf
import numpy as np


class AngularAcceptance:
    """
    Angular acceptance: ε(Ω) = Σ_{k=1}^{10} c_k * f_k(Ω)
    
    The 10 basis functions f_k correspond to the angular terms
    used in LHCb-ANA-2019-032, expressed in terms of helicity angles:
        - θ_K: K* helicity angle (K+ vs Bs rest frame)
        - θ_μ: J/ψ helicity angle (μ+ vs J/ψ rest frame)
        - φ:   azimuthal angle between decay planes

    Coordinate mapping in tf-pwa (from fit_conv.py AnglesPreprocessor):
        data["p4"] shape (N, 3) stores:
            column 0: theta_μ  (arccos(helcosthetaL))
            column 1: theta_K  (arccos(helcosthetaK))
            column 2: phi_K    (helphi)
    
    Per-candidate calculation:
        For each event, compute all 10 basis functions f_k(Ω),
        multiply by the corresponding coefficients c_k,
        and sum to get the acceptance value.
    """
    
    def __init__(self, omega_unbiased=None, omega_biased=None):
        """
        Initialize AngularAcceptanceOmega with omega values.
        
        Args:
            omega_unbiased: array of 10 omega values [omega1..omega10] for unbiased trigger.
                            omega1=1.0 (fixed). Default: all 1.0 (flat acceptance)
            omega_biased: array of 10 omega values [omega1..omega10] for biased trigger.
                          omega1=1.0 (fixed). Default: all 1.0 (flat acceptance)
        """
        if omega_unbiased is None:
            omega_unbiased = np.ones(10, dtype=np.float64)
        else:
            omega_unbiased = np.array(omega_unbiased, dtype=np.float64)
        omega_unbiased[0] = 1.0
        
        if omega_biased is None:
            omega_biased = np.ones(10, dtype=np.float64)
        else:
            omega_biased = np.array(omega_biased, dtype=np.float64)
        omega_biased[0] = 1.0
        
        self.omega_unbiased = omega_unbiased
        self.omega_biased = omega_biased
        
        self.c_unbiased = self._compute_c_coefficients(omega_unbiased)
        self.c_biased = self._compute_c_coefficients(omega_biased)
        
        self.norm_unbiased = self._compute_normalization(self.c_unbiased)
        self.norm_biased = self._compute_normalization(self.c_biased)
        
        print("AngularAcceptanceOmega initialized:")
        print(f"  Unbiased c coefficients: {self.c_unbiased}")
        print(f"  Unbiased normalization factor: {self.norm_unbiased}")
        print(f"  Biased c coefficients: {self.c_biased}")
        print(f"  Biased normalization factor: {self.norm_biased}")
    
    def _compute_c_coefficients(self, omega):
        """
        Compute c_k coefficients from omega_k values using Solve-8 formulas.
        
        Block A (4×4 inverse matrix solution):
            c1 = (1/π)(1755/512 ω1 + 135/128 ω2 + 135/128 ω3 - 2565/512 ω7)
            c2 = (1/π)(135/128 ω1 + 315/64 ω2 - 135/64 ω3 - 405/128 ω7)
            c3 = (1/π)(135/128 ω1 - 135/64 ω2 + 315/64 ω3 - 405/128 ω7)
            c7 = (1/π)(-2565/512 ω1 - 405/128 ω2 - 405/128 ω3 + 6075/512 ω7)
        
        Blocks B, C, D (direct division by diagonal elements):
            c4 = 225ω4/(64π)
            c5 = 225ω5/(32π)
            c6 = 225ω6/(32π)
            c8 = 135ω8/(32π)
            c9 = 135ω9/(32π)
            c10 = 135ω10/(256π)
        
        Args:
            omega: array of 10 omega values [omega1..omega10]
            
        Returns:
            array of 10 c coefficients [c1..c10]
        """
        omega_1, omega_2, omega_3, omega_4, omega_5, omega_6, omega_7, omega_8, omega_9, omega_10 = omega
        
        pi = np.pi
        
        c1 = (1.0 / pi) * (1755.0 / 512.0 * omega_1 + 135.0 / 128.0 * omega_2 + 135.0 / 128.0 * omega_3 - 2565.0 / 512.0 * omega_7)
        c2 = (1.0 / pi) * (135.0 / 128.0 * omega_1 + 315.0 / 64.0 * omega_2 - 135.0 / 64.0 * omega_3 - 405.0 / 128.0 * omega_7)
        c3 = (1.0 / pi) * (135.0 / 128.0 * omega_1 - 135.0 / 64.0 * omega_2 + 315.0 / 64.0 * omega_3 - 405.0 / 128.0 * omega_7)
        c7 = (1.0 / pi) * (-2565.0 / 512.0 * omega_1 - 405.0 / 128.0 * omega_2 - 405.0 / 128.0 * omega_3 + 6075.0 / 512.0 * omega_7)
        
        c4 = (225.0 / (64.0 * pi)) * omega_4
        c5 = (225.0 / (32.0 * pi)) * omega_5
        c6 = (225.0 / (32.0 * pi)) * omega_6
        c8 = (135.0 / (32.0 * pi)) * omega_8
        c9 = (135.0 / (32.0 * pi)) * omega_9
        c10 = (135.0 / (256.0 * pi)) * omega_10
        
        return np.array([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10], dtype=np.float64)
    
    def _compute_normalization(self, c):
        """
        Compute normalization factor N for ε(Ω) to ensure <ε(Ω)> = 1 over uniform phase space.
        
        According to Solve-8 and empirical verification:
            N = 8π / ∑ₖ cₖ ∫ fₖ dΩ
        
        Based on numerical integration over uniform phase space, the basis function integrals are:
            <f₁> = ∫ f₁ dΩ / 8π ≈ 0.222
            <f₂> = ∫ f₂ dΩ / 8π ≈ 0.222
            <f₃> = ∫ f₃ dΩ / 8π ≈ 0.222
            <f₇> = ∫ f₇ dΩ / 8π ≈ 0.222
            <f₄..6, 8..10> ≈ 0 (odd symmetry)
        
        Therefore:
            N = 8π / (c₁*∫f₁ + c₂*∫f₂ + c₃*∫f₃ + c₇*∫f₇)
              = 1 / (<f₁>*c₁ + <f₂>*c₂ + <f₃>*c₃ + <f₇>*c₇)
        
        Using exact analytical values:
            ∫ f₁ dΩ = ∫ cos²θ_K sin²θ_L dΩ = (2/3) * (2/3) * 2π = 8π/9
            ∫ f₂ dΩ = ∫ 0.5 sin²θ_K (1 - cos²φ sin²θ_L) dΩ = 8π/9
            ∫ f₃ dΩ = ∫ 0.5 sin²θ_K (1 - sin²φ sin²θ_L) dΩ = 8π/9
            ∫ f₇ dΩ = ∫ (1/3) sin²θ_L dΩ = 8π/9
            
            N = 8π / (c₁*(8π/9) + c₂*(8π/9) + c₃*(8π/9) + c₇*(8π/9))
              = 9 / (c₁ + c₂ + c₃ + c₇)
        
        Args:
            c: array of 10 c coefficients [c1..c10]
            
        Returns:
            normalization factor N
        """
        denominator = c[0] + c[1] + c[2] + c[6]
        
        if denominator == 0.0:
            raise ValueError("Denominator for normalization factor is zero!")
        
        N = 4.5 / denominator
        
        return N
    
    def eval_basis(self, cos_theta_K, cos_theta_L, phi):
        """
        Calculate 10 angular basis functions f_k(Ω) for per-candidate evaluation.
        
        Following LHCb-ANA-2019-032 convention:
        f_k are the angular functions of the transversity basis
        that multiply the time-dependent terms.
        
        Args:
            cos_theta_K: cos(θ_K) for each candidate (N,)
            cos_theta_L: cos(θ_μ) for each candidate (N,)
            phi: φ for each candidate (N,)
            
        Returns:
            tensor of shape (n_events, 10) - the basis functions f_k for each event
        """
        cos_theta_K = np.array(cos_theta_K, dtype=np.float64)
        cos_theta_L = np.array(cos_theta_L, dtype=np.float64)
        phi = np.array(phi, dtype=np.float64)

        ctK = cos_theta_K
        ctL = cos_theta_L
        stK = tf.sqrt(tf.maximum(1.0 - ctK ** 2, 1e-12))
        stL = tf.sqrt(tf.maximum(1.0 - ctL ** 2, 1e-12))
        sphi = tf.sin(phi)
        cphi = tf.cos(phi)

        f = tf.stack(
            [
                ctK ** 2 * stL ** 2,                         # f1: |A0|^2
                0.5 * stK ** 2 * (1.0 - cphi ** 2 * stL ** 2),  # f2: |A_parallel|^2
                0.5 * stK ** 2 * (1.0 - sphi ** 2 * stL ** 2),  # f3: |A_perp|^2
                stK ** 2 * stL ** 2 * sphi * cphi,            # f4: A_perp A_parallel
                np.sqrt(2.0) * stK * ctK * stL * ctL * cphi,  # f5: A0 A_parallel
                -np.sqrt(2.0) * stK * ctK * stL * ctL * sphi, # f6: A0 A_perp
                (1.0 / 3.0) * stL ** 2,                       # f7: |AS|^2
                2.0 / np.sqrt(6.0) * stK * stL * ctL * cphi,   # f8: AS A_parallel
                -2.0 / np.sqrt(6.0) * stK * stL * ctL * sphi,  # f9: AS A_perp
                2.0 / np.sqrt(3.0) * ctK * stL ** 2,            # f10: AS A0
            ],
            axis=-1,
        )
        
        return f
    
    def __call__(self, cos_theta_K, cos_theta_L, phi, trigger=None):
        """
        Evaluate ε(Ω) = N * Σ c_k * f_k(Ω) for each candidate.
        
        According to Solve-8 formula:
            ε(Ω) = Σ_{k=1}^{10} c_k * f_k(Ω)
        
        With normalization factor N to ensure <ε(Ω)> = 1 over uniform phase space:
            N = 9 / (c₁ + 0.5*c₂ + 0.5*c₃ + c₇)
        
        Args:
            cos_theta_K: cos(θ_K) for each candidate (N,)
            cos_theta_L: cos(θ_μ) for each candidate (N,)
            phi: φ for each candidate (N,)
            trigger: Optional trigger flag array (N,), 0=unbiased, 1=biased
            
        Returns:
            Tensor of shape (N,) with per-candidate angular acceptance values
        """
        f = self.eval_basis(cos_theta_K, cos_theta_L, phi)
        
        if trigger is not None:
            trigger_np = (
                trigger.numpy() if hasattr(trigger, "numpy") else np.array(trigger)
            )
            c = np.where(
                trigger_np[..., None] == 1,
                self.c_biased,
                self.c_unbiased,
            )
            norm = np.where(trigger_np == 1, self.norm_biased, self.norm_unbiased)
        else:
            c = np.broadcast_to(self.c_unbiased, f.shape)
            norm = self.norm_unbiased
        
        unnormalized_result = np.sum(f.numpy() * c, axis=-1)
        result = unnormalized_result * norm
        
        return tf.constant(result, dtype=f.dtype)
    
    @classmethod
    def from_json(cls, json_path, year):
        """
        Load omega values from JSON file and create AngularAcceptanceOmega instance.
        
        JSON file format:
        {
            "year": {
                "OmegaUnbiased": [
                    {"Name": "omega1", "Value": 1.0, "Error": 0.0},
                    ...
                    {"Name": "omega10", "Value": x.x, "Error": x.x}
                ],
                "OmegaBiased": [
                    {"Name": "omega1", "Value": 1.0, "Error": 0.0},
                    ...
                    {"Name": "omega10", "Value": x.x, "Error": x.x}
                ]
            }
        }
        
        Args:
            json_path: path to JSON file
            year: year to load (integer or string)
            
        Returns:
            AngularAcceptanceOmega instance
        """
        import json
        with open(json_path) as f:
            data = json.load(f)
        
        year_str = str(year)
        if year_str not in data:
            raise KeyError(f"Year {year} not found in JSON file")
        
        year_data = data[year_str]
        
        omega_unbiased = []
        omega_biased = []
        
        if "OmegaUnbiased" in year_data:
            for i in range(1, 11):
                found = False
                for item in year_data["OmegaUnbiased"]:
                    if item.get("Name") == f"omega{i}":
                        omega_unbiased.append(item["Value"])
                        found = True
                        break
                if not found:
                    omega_unbiased.append(1.0 if i == 1 else 0.0)
        else:
            omega_unbiased = np.ones(10, dtype=np.float64)
        
        if "OmegaBiased" in year_data:
            for i in range(1, 11):
                found = False
                for item in year_data["OmegaBiased"]:
                    if item.get("Name") == f"omega{i}":
                        omega_biased.append(item["Value"])
                        found = True
                        break
                if not found:
                    omega_biased.append(1.0 if i == 1 else 0.0)
        else:
            omega_biased = np.ones(10, dtype=np.float64)
        
        return cls(omega_unbiased=omega_unbiased, omega_biased=omega_biased)
