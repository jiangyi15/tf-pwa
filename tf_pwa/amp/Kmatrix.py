import sympy as sym
import tensorflow as tf

from tf_pwa.amp import Particle
from tf_pwa.amp import get_relative_p as get_relative_p2
from tf_pwa.amp import register_particle
from tf_pwa.breit_wigner import get_bprime_coeff


def get_relative_p(m0, m1, m2):
    a = m0 ** 2 - (m1 + m2) ** 2
    b = m0 ** 2 - (m1 - m2) ** 2
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
    return sym.sqrt(frac * z ** l / d)


def Bl(l, q, q0, d=3):
    return Fb(l, q, d) / Fb(l, q0, d)


def KMatrix_single(n_pole, m1, m2, l, d=3, bkg=0):
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
        denominator = mi[i] ** 2 - m ** 2
        tmp = tmp / denominator
        K = tmp + K

    P = 0
    for i in range(n_pole):
        # remove Bl function in gls
        tmp = betai[i] * mi[i] * g0i[i]  # * Bl(l, p, p0[i], d)
        P = tmp / (mi[i] * mi[i] - m * m) + P
    P = P + bkg  #  + sym.Symbol("d", complex=True)
    return P / (sym.S(1) - sym.I * K)


@register_particle("KMatrixSingleChannel")
class KmatrixSingleChannelParticle(Particle):
    def __init__(self, *args, **kwargs):
        self.d = 3
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

        self.symbol = sym.simplify(
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
