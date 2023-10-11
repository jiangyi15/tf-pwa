import sympy as sym
import tensorflow as tf

from tf_pwa.amp import HelicityDecay, Particle
from tf_pwa.amp import get_relative_p as get_relative_p2
from tf_pwa.amp import register_decay, register_particle
from tf_pwa.breit_wigner import Bprime_q2, get_bprime_coeff


def get_relative_p(m0, m1, m2):
    a = m0**2 - (m1 + m2) ** 2
    b = m0**2 - (m1 - m2) ** 2
    return sym.sqrt(a * b) / m0 / sym.S(2)


def Fb(l, q, d=3):
    if l == 0:
        return sym.S(1)
    z = (q * d) ** 2
    coef = get_bprime_coeff(l)
    frac = sum(coef)
    d = 0
    for i, k in enumerate(coef):
        d = d + k * z ** (l - i)
    return sym.sqrt(frac * z**l / d)


def Bl(l, q, q0, d=3):
    return Fb(l, q, d) / Fb(l, q0, d)


def KMatrix_single(n_pole, m1, m2, l, d=3, bkg=0, Kb=0):
    m = sym.Symbol("m", positive=True)
    mi = [sym.Symbol(f"m{i}", positive=True) for i in range(n_pole)]
    g0i = [sym.Symbol(f"g{i}", real=True) for i in range(n_pole)]
    betai = [
        sym.Symbol(f"alpha{i}", real=True)
        + sym.I * sym.Symbol(f"beta{i}", real=True)
        for i in range(n_pole)
    ]
    p = sym.Symbol(f"p", positive=True)  # get_relative_p(m, m1, m2) #
    p0 = [
        sym.Symbol(f"p0{i}", positive=True) for i in range(n_pole)
    ]  # [get_relative_p(m, m1, m2) for m in mi] #
    rho = sym.S(2) * p / m
    K = 0
    for i in range(n_pole):
        tmp = (
            mi[i]
            * g0i[i]
            * (mi[i] / m)
            * (p / p0[i])
            * Bl(l, p, p0[i], d) ** 2
        )
        denominator = mi[i] ** 2 - m**2
        tmp = tmp / denominator
        K = tmp + K
    k = K + Kb

    P = 0
    for i in range(n_pole):
        # remove Bl function in gls
        tmp = betai[i] * mi[i] * g0i[i]  # * Bl(l, p, p0[i], d)
        P = tmp / (mi[i] * mi[i] - m * m) + P
    P = P + bkg  #  + sym.Symbol("d", complex=True)
    return P / (sym.S(1) - sym.I * K)


@register_particle("KMatrixSingleChannel")
class KmatrixSingleChannelParticle(Particle):
    """
    K matrix model for single channel multi pole.

    .. math::
        K = \\sum_{i} \\frac{m_i \\Gamma_i(m)}{m_i^2 - m^2 }

    .. math::
        P = \\sum_{i} \\frac{\\beta_i m_0 \\Gamma_0 }{ m_i^2 - m^2}

    the barrier factor is included in gls

    .. math::
        R(m) = (1-iK)^{-1} P

    requird :code:`mass_list: [pole1, pole2]` and :code:`width_list: [width1, width2]`.

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> from tf_pwa.utils import plot_particle_model
        >>> axis = plot_particle_model("KMatrixSingleChannel", {"mass_list": [0.5, 0.6], "width_list": [0.03, 0.05], "mass": 0.5, "m1": 0.1, "m2": 0.1},
        ...  {"R_BC_beta1r": 1.,"R_BC_beta2r": 0.0, "R_BC_beta1i": 0.,"R_BC_beta2i": 0.0})
        ...
        >>> axis = plot_particle_model("KMatrixSingleChannel", {"mass_list": [0.5, 0.6], "width_list": [0.03, 0.05], "mass": 0.5, "m1": 0.1, "m2": 0.1},
        ...  {"R_BC_beta1r": 0.,"R_BC_beta2r": 1.0, "R_BC_beta1i": 0.,"R_BC_beta2i": 0.0}, axis=axis)
        ...
        >>> axis = plot_particle_model("KMatrixSingleChannel", {"mass_list": [0.5, 0.6], "width_list": [0.03, 0.05], "mass": 0.5},
        ...  {"R_BC_beta1r": 1.,"R_BC_beta2r": 1.0, "R_BC_beta1i": 0.,"R_BC_beta2i": 0.0}, axis=axis)
        ...
        >>> _ = axis[3].legend([" $\\\\beta_1=1$ ", " $\\\\beta_2=1$ ", " $\\\\beta_1=\\\\beta_2=1$ "], fontsize=8)

    """

    def __init__(self, *args, **kwargs):
        self.d = 3
        self.m1 = None
        self.m2 = None
        super().__init__(*args, **kwargs)
        self.n_pole = len(self.mass_list)

    def init_params(self):
        self.mi = []
        self.gi = []
        self.beta = []

        assert self.n_pole > 0
        if self.bw_l is None:
            decay = self.decay[0]
            self.bw_l = min(decay.get_l_list())
        if self.m1 is None:
            self.m1 = float(self.decay[0].outs[0].get_mass())
        if self.m2 is None:
            self.m2 = float(self.decay[0].outs[1].get_mass())

        self.symbol = sym.together(
            KMatrix_single(self.n_pole, self.m1, self.m2, self.bw_l, self.d)
        )
        self.function = opt_lambdify(
            self.symbol.free_symbols,
            (sym.re(self.symbol), sym.im(self.symbol)),
            modules="tensorflow",
        )

        for i in range(self.n_pole):
            self.mi.append(
                self.add_var(f"mass{i+1}", value=self.mass_list[i], fix=True)
            )
            self.gi.append(
                self.add_var(f"width{i+1}", value=self.width_list[i], fix=True)
            )
            self.beta.append(self.add_var(f"beta{i+1}", is_complex=True))
        self.beta[0].fixed(1.0)

    def get_mi(self):
        return [self.mi[i]() for i in range(self.n_pole)]

    def get_gi(self):
        return [self.gi[i]() for i in range(self.n_pole)]

    def get_beta(self):
        ret_r = [tf.math.real(self.beta[i]()) for i in range(self.n_pole)]
        ret_i = [tf.math.imag(self.beta[i]()) for i in range(self.n_pole)]
        return ret_r, ret_i

    def get_amp(self, *args, **kwargs):
        m = args[0]["m"]
        mi = self.get_mi()
        gi = self.get_gi()
        beta_r, beta_i = self.get_beta()
        params = {"m": m}
        params["p"] = get_relative_p2(m, self.m1, self.m2)
        for i in range(self.n_pole):
            params[f"m{i}"] = mi[i]
            params[f"g{i}"] = gi[i]
            params[f"alpha{i}"] = beta_r[i]
            params[f"beta{i}"] = beta_i[i]
            params[f"p0{i}"] = get_relative_p2(mi[i], self.m1, self.m2)

        ret_r, ret_i = self.function(**params)
        return tf.complex(ret_r, ret_i)


def opt_lambdify(args, expr, **kwargs):
    subs, ret = sym.cse(expr)
    # print(subs, ret)
    head = "def _generate_fun({}):\n".format(",".join(str(i) for i in args))
    for a, b in subs:
        head += "  {} = {}\n".format(a, b).replace("sqrt", "tf.sqrt")
    head += "  return {}\n".format(ret)
    var = {}
    exec(head, globals(), var)
    return var["_generate_fun"]


@register_decay("LS-decay-Kmatrix")
class ParticleDecayLSKmatrix(HelicityDecay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_params(self):
        self.d = 3.0

    def get_ls_amp(self, data, data_p, **kwargs):
        if "|q|2" in data:
            q = data["|q|2"]
        else:
            q = self.get_relative_momentum2(data_p, True)
            data["|q|2"] = q

        return self.core.get_ls_amp(data_p[self.core]["m"])


@register_particle("KMatrixSplitLS")
class KmatrixSplitLSParticle(Particle):
    """
    K matrix model for single channel multi pole and the same channel with different (l, s) coupling.

    .. math::
        K_{a,b} = \\sum_{i} \\frac{m_i \\sqrt{\\Gamma_{a,i}(m)\\Gamma_{b,i}(m)}}{m_i^2 - m^2 }

    .. math::
        P_{b} = \\sum_{i} \\frac{\\beta_i m_0 \\Gamma_{b,i0} }{ m_i^2 - m^2}

    the barrier factor is included in gls

    .. math::
        R(m) = (1-iK)^{-1} P

    """

    def __init__(self, *args, **kwargs):
        self.d = 3
        self.m1 = None
        self.m2 = None
        super().__init__(*args, **kwargs)
        self.decay_params = kwargs.get("decay_params", {})
        self.decay_params = {
            "model": "LS-decay-Kmatrix",
            **self.decay_params,
        }
        self.n_pole = len(self.mass_list)
        self.ls_list = None

    def init_params(self):
        self.mi = []
        self.gi = []
        self.gi_frac = []
        self.beta = []
        assert self.n_pole > 0

        decay = self.decay[0]
        self.ls_list = decay.get_l_list()

        for i in range(self.n_pole):
            self.mi.append(
                self.add_var(f"mass{i+1}", value=self.mass_list[i], fix=True)
            )
            self.gi.append(
                self.add_var(f"width{i+1}", value=self.width_list[i], fix=True)
            )
            self.beta.append(self.add_var(f"beta{i+1}", is_complex=True))

        for j in range(self.n_pole):
            self.gi_frac.append([])
            for i, _ in enumerate(self.ls_list[1:]):
                tmp = self.add_var(f"theta{j}_{i}")
                self.gi_frac[j].append(tmp)
        self.beta[0].fixed(1.0)
        if self.m1 is None:
            assert len(self.decay) == 1
            self.m1 = self.decay[0].outs[0].get_mass()
        if self.m2 is None:
            self.m2 = self.decay[0].outs[1].get_mass()

    def get_mi(self):
        return [self.mi[i]() for i in range(self.n_pole)]

    def get_gi(self):
        return [self.gi[i]() for i in range(self.n_pole)]

    def get_beta(self):
        ret_r = [tf.math.real(self.beta[i]()) for i in range(self.n_pole)]
        ret_i = [tf.math.imag(self.beta[i]()) for i in range(self.n_pole)]
        return ret_r, ret_i

    def get_gi_frac(self):
        ret = []
        for i in self.gi_frac:
            if len(i) == 0:
                tmp = [1]
            else:
                t = i[0]()
                a, b = tf.cos(t), tf.sin(t)
                tmp = [a]
                for j in i[1:]:
                    a = b * tf.cos(j())
                    b = b * tf.sin(j())
                    tmp.append(a)
                tmp.append(b)
            ret.append(tmp)
        return ret

    def get_amp(self, *args, **kwargs):
        m = args[0]["m"]
        zeros = tf.zeros_like(m)
        ones = tf.ones_like(m)
        return tf.complex(ones, zeros)

    def get_ls_amp(self, m):
        zeros = tf.zeros_like(m)
        ones = tf.ones_like(m)
        mi = self.get_mi()
        gi = self.get_gi()
        beta_r, beta_i = self.get_beta()
        _epsilon = 1e-4
        p = get_relative_p2(m, self.m1, self.m2)
        # print("p", p)
        K = [[0 for i in self.ls_list] for j in self.ls_list]
        p0 = [
            get_relative_p2(mi[i], self.m1, self.m2)
            for i in range(self.n_pole)
        ]
        # print("p0", p0)
        gi_frac = self.get_gi_frac()

        dm = []

        den_prod = tf.complex(ones, zeros)
        for k in range(self.n_pole):
            tmp = tf.complex(mi[k] ** 2 - m**2, -_epsilon * ones)
            dm.append(tmp)
            den_prod = den_prod * dm[k]

        complex_i = tf.complex(zeros, ones)
        for i, l1 in enumerate(self.ls_list):
            for j, l2 in enumerate(self.ls_list):
                for k in range(self.n_pole):
                    den_prod_i = tf.complex(ones, zeros)
                    for l in range(self.n_pole):
                        if l != k:
                            den_prod_i = den_prod_i * dm[l]

                    rho = tf.sqrt(tf.complex(p / m * mi[k] / p0[k], zeros))
                    # print(den_prod_i, den_prod, rho)
                    # print(p0, k)
                    bf1 = (p / p0[k]) ** (l1 / 2) * Bprime_q2(
                        l1, p, p0[k], self.d
                    )
                    bf2 = (p / p0[k]) ** (l2 / 2) * Bprime_q2(
                        l2, p, p0[k], self.d
                    )

                    # print(l1, l2, bf1, bf2)

                    K[i][j] = K[i][
                        j
                    ] - complex_i * rho * den_prod_i * tf.complex(
                        gi[k] * gi_frac[k][i] * gi_frac[k][j] * bf1 * bf2,
                        zeros,
                    )
                if i == j:
                    K[i][j] = den_prod + K[i][j]

        # det = K[0][0] * K[1][1] - K[0][1]*K[1][0]
        K_inv = tf.linalg.inv(
            tf.stack([tf.stack(i, axis=-1) for i in K], axis=-2)
        )  # [[K[1][1]/det, -K[1][0]/det],[-K[0][1]/det, K[0][0]/det]]
        # K_inv = tf.stack([tf.stack(i, axis=-1) for i in K_inv], axis=-1)

        # print("num of nan", tf.reduce_sum(tf.math.is_nan(K_inv)))

        P = [0 for i in self.ls_list]
        for i, l in enumerate(self.ls_list):
            for k in range(self.n_pole):
                den_prod_i = tf.complex(ones, zeros)
                for l in range(self.n_pole):
                    if l != k:
                        den_prod_i = den_prod_i * dm[l]

                # print(den_prod_i, den_prod)
                P[i] = P[i] + den_prod_i * tf.complex(
                    mi[k] * gi[k] * gi_frac[k][i], zeros
                ) * tf.complex(
                    beta_r[k], zeros
                )  #  beta_i[k])

        # E = tf.linalg.diag(tf.ones((len(self.ls_list,)), dtype=tf.complex128)* tf.complex(den_prod, zeros)[:,None])

        # print("den", E - tf.stack(K, axis=-2))
        # print(tf.linalg.det(E - tf.stack(K, axis=-2)))
        # K_inv = tf.linalg.inv(E - tf.stack(K, axis=-2))
        # print("K_inv", K_inv)
        # print(P)
        ret = tf.reduce_sum(K_inv * tf.stack(P, axis=-1)[:, None], axis=1)
        return ret
