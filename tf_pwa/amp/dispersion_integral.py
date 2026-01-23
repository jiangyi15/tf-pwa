"""

The dispersion relation is come from the theory for functions of complex variables.
When a function :math:`f(z)` is analytic in the real axis and vanish when :math:`z \\rightarrow \\infty`,
The function has the property that the integration over the edge of the upper half plane is zero,

.. math::
    \\int_C  \\frac{f(z)}{z -z_0 } \\mathrm{d} z = \\lim_{\\epsilon\\rightarrow 0}\\left[\\int_{-\\infty}^{z_0-\\epsilon}\\frac{f(z)}{z - z_0} \\mathrm{d} z + \\int_{z_0+\\epsilon}^{\\infty} \\frac{f(z)}{z -z_0 } \\mathrm{d} z + \\int_{z_0-\\epsilon}^{z_0+\\epsilon}\\frac{f(z)}{z -z_0 } \\mathrm{d} z \\right]  = 0

.. plot::

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> _ = plt.plot([1.0], [0.0], marker="o")
    >>> phi = np.linspace(0,np.pi)
    >>> _ = plt.plot(1.0 + 0.1*np.cos(phi), 0.1*np.sin(phi), c="C1")
    >>> _ = plt.plot(-3 * np.cos(phi), 3 *np.sin(phi), c="C2")
    >>> _ = plt.plot([-3, 0.9], [0,0], c="C3")
    >>> _ = plt.plot([1.1, 3], [0,0], c="C3")
    >>> _ = plt.xticks([-3, 0, 1, 3], labels=["$-\\infty$", "0", "z0", "$-\\infty$"])
    >>> _ = plt.yticks([])


The integration suround the pole :math:`z_0` contibute a residue term.

.. math::
    i \\pi f(z_0) = P\\int_{-\\infty}^{+\\infty}\\frac{f(z)}{z - z_0} \\mathrm{d} z = \\lim_{\\epsilon \\rightarrow 0}\\left[\\int_{-\\infty}^{z_0-\\epsilon}\\frac{f(z)}{z - z_0} \\mathrm{d} z + \\int_{z_0+\\epsilon}^{\\infty} \\frac{f(z)}{z -z_0 } \\mathrm{d} z\\right]


The physical amplitude have the same property after some subtraction of infinity.

.. math::
    Re f(s) - Re f(s_0) = \\frac{1}{\\pi}P\\int_{s_{th}}^{\\infty} \left[\\frac{Im f(s')}{s' - s} - \\frac{Im f(s')}{s' - s_0}\\right]\mathrm{d} s' = \\frac{(s-s_0)}{\\pi}P\\int_{s_{th}}^{\\infty} \\frac{Im f(s')}{(s' - s)(s' - s_0)}\\mathrm{d} s'

Sometimes, additional substrction will be used to make sure that the intergration is finity.

.. math::
    Re f(s) - Re \\left[\\sum_{k=0}^{n-1}\\frac{(s-s_0)^k f^{(k)}(s_0)}{k!}\\right] = \\frac{(s-s_0)^n}{\\pi}P\\int_{s_{th}}^{\\infty} \\frac{Im f(s')}{(s' - s)(s' - s_0)^n}\\mathrm{d} s'

More detials can be found in `Dispersion Relation Integrals <https://analyticphysics.com/S-Matrix%20Theory/Dispersion%20Relation%20Integrals.htm>`_.

"""

import numpy as np
import tensorflow as tf

from tf_pwa.amp.core import Particle, register_particle
from tf_pwa.breit_wigner import chew_mandelstam, complex_q
from tf_pwa.data import data_to_numpy


def build_integral(fun, s_range, s_th, N=1001, add_tail=True, method="tf"):
    """

    .. math::
       F(s) = P\\int_{s_{th}}^{\\infty} \\frac{f(s')}{s'-s} \\mathrm{d} s'
            = \\int_{s_{th}}^{s- \\epsilon} \\frac{f(s')}{s'-s} \\mathrm{d} s' + \\int_{s + \\epsilon}^{s_{max} } \\frac{f(s')}{s'-s} \\mathrm{d} s' + \\int_{s_{max}}^{\\infty} \\frac{f(s')}{s'-s} \\mathrm{d} s'

    It require same :math:`\\epsilon` for :math:`s- \\epsilon` and :math:`s- \\epsilon` to get the Cauchy Principal Value. We used bin center to to keep the same :math:`\\epsilon` from left and right bound.

    """
    if method == "scipy":
        return build_integral_scipy(fun, s_range, s_th, N=N, add_tail=add_tail)
    else:
        return build_integral_tf(fun, s_range, s_th, N=N, add_tail=add_tail)


def build_integral_scipy(
    fun, s_range, s_th, N=1001, add_tail=True, _epsilon=1e-6
):
    """

    .. math::
       F(s) = P\\int_{s_{th}}^{\\infty} \\frac{f(s')}{s'-s} \\mathrm{d} s'
            = \\int_{s_{th}}^{s- \\epsilon} \\frac{f(s')}{s'-s} \\mathrm{d} s' + \\int_{s + \\epsilon}^{s_{max} } \\frac{f(s')}{s'-s} \\mathrm{d} s' + \\int_{s_{max}}^{\\infty} \\frac{f(s')}{s'-s} \\mathrm{d} s'

    It require same :math:`\\epsilon` for :math:`s- \\epsilon` and :math:`s- \\epsilon` to get the Cauchy Principal Value. We used bin center to to keep the same :math:`\\epsilon` from left and right bound.

    """
    from scipy.integrate import quad

    s_min, s_max = s_range
    x = np.linspace(s_min, s_max, N)

    ret = []

    def f(s):
        with tf.device("CPU"):
            y = data_to_numpy(fun(s))
        return y

    for xi in x:
        if xi < s_th:
            y, e = quad(lambda s: f(s) / (s - xi), s_th + _epsilon, np.inf)
        else:
            y1, e1 = quad(lambda s: f(s) / (s - xi), s_th, xi - _epsilon)
            y2, e2 = quad(
                lambda s: f(s) / (s - xi), xi + _epsilon, s_max + 0.1
            )
            y3, e2 = quad(lambda s: f(s) / (s - xi), s_max + 0.1, np.inf)
            y = y1 + y2 + y3
        ret.append(y)
    return x, np.stack(ret)


def build_integral_tf(fun, s_range, s_th, N=1001, add_tail=True):
    """

    .. math::
       I(s) = P\\int_{s_{th}}^{\\infty} \\frac{f(s')}{s'-s} \\mathrm{d} s'
            = \\int_{s_{th}}^{s- \\epsilon} \\frac{f(s')}{s'-s} \\mathrm{d} s' + \\int_{s + \\epsilon}^{s_{max} } \\frac{f(s')}{s'-s} \\mathrm{d} s' + \\int_{s_{max}}^{\\infty} \\frac{f(s')}{s'-s} \\mathrm{d} s'

    It require same :math:`\\epsilon` for :math:`s- \\epsilon` and :math:`s- \\epsilon` to get the Cauchy Principal Value. We used bin center to to keep the same :math:`\\epsilon` from left and right bound.

    """
    s_min, s_max = s_range
    delta = (s_min - s_max) / (N - 1)
    x = np.linspace(s_min - delta / 2, s_max + delta / 2, N + 1)
    shift = (s_th - s_min) % delta
    x = x + shift * delta
    x_center = (x[1:] + x[:-1]) / 2
    int_x = x[x > s_th + delta / 4]
    fx = fun(int_x) / (int_x - x_center[:, None])
    int_f = tf.reduce_mean(fx, axis=-1) * (int_x[-1] - int_x[0] + delta)
    if add_tail:
        int_f = int_f + build_integral_tail(
            fun, x_center, x[-1] + delta / 2, s_th, N
        )
    return x_center, int_f


def build_integral_tail(fun, x_center, tail, s_th, N=1001, _epsilon=1e-9):
    """

    Integration of the tail parts using tan transfrom.

    .. math::
       \\int_{s_{max}}^{\\infty} \\frac{f(s')}{s'-s} \\mathrm{d} s' = \\int_{\\arctan s_{max}}^{\\frac{\\pi}{2}} \\frac{f(\\tan x)}{\\tan x-s} \\frac{\\mathrm{d} \\tan x}{\\mathrm{d} x} \\mathrm{d} x

    """
    x_min, x_max = np.arctan(tail), np.pi / 2
    delta = (x_min - x_max) / N
    x = np.linspace(x_min + delta / 2, x_max - delta / 2, N)
    tanx = tf.tan(x)
    dtanxdx = 1 / tf.cos(x) ** 2
    fx = fun(tanx) / (tanx - x_center[:, None]) * dtanxdx
    int_f = tf.reduce_mean(fx, axis=-1) * (np.pi / 2 - np.arctan(tail))
    return int_f


class LinearInterpFunction:
    "class for linear interpolation"

    def __init__(self, x_range, y):
        x_min, x_max = x_range
        N = y.shape[-1]
        self.x_min = x_min
        self.x_max = x_max
        self.delta = (self.x_max - self.x_min) / (N - 1)
        self.N = N
        self.y = y

    def __call__(self, x):
        diff = (x - self.x_min) / self.delta
        idx0 = diff // 1.0
        idx = tf.cast(idx0, tf.int32)
        left = tf.gather(self.y, idx)
        right = tf.gather(self.y, idx + 1)
        k = diff - idx0
        return (1 - k) * left + k * right


class DispersionIntegralFunction(LinearInterpFunction):
    "class for interpolation of dispersion integral."

    def __init__(self, fun, s_range, s_th, N=1001, method="tf"):
        self.fun = fun
        x_center, y = build_integral(fun, s_range, s_th, N, method=method)
        super().__init__((x_center[0], x_center[-1]), y)


@register_particle("DI")
class DispersionIntegralParticle(Particle):
    """

    Dispersion Integral model. In the model a linear interpolation is used to avoid integration every times in fitting. No paramters are allow in the integration.

    .. math::
        f(s) = \\frac{1}{m_0^2 - s - \\sum_{i} g_i^2 [Re\\Pi_i(s) -Re\\Pi_i(m_0^2) + i Im \\Pi_i(s)] }

    where :math:`Im \\Pi_i(s)=\\rho_i(s)n_i^2(s)`, :math:`n_i(s)={q}^{l} {B_l'}(q,1/d, d)`.

    The real parts of :math:`\\Pi(s)` is defined using the dispersion intergral

    .. math::

        Re \\Pi_i(s) = \\frac{\\color{red}(s-s_{0,i})}{\\pi} P \\int_{s_{th,i}}^{\\infty} \\frac{Im \\Pi_i(s')}{(s' - s){\\color{red} (s'-s_{0,i})}} \\mathrm{d} s'

    By default, :math:`s_{0,i}=0`, it can be change to other value though option `s0: value`. `value=sth` for :math:`s_{th,i}`.

    .. note::

        Small `int_N` will have bad precision.

    The shape of :math:`\\Pi(s)` and comparing to Chew-Mandelstam function :math:`\\Sigma(s)`

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> import tensorflow as tf
        >>> from tf_pwa.amp.dispersion_integral import DispersionIntegralParticle, chew_mandelstam, complex_q
        >>> plt.clf()
        >>> m = np.linspace(0.6, 1.6, 1001)
        >>> s = m * m
        >>> M_K = 0.493677
        >>> M_eta = 0.547862
        >>> M_pi = 0.1349768
        >>> p = DispersionIntegralParticle("A0_980_DI", mass_range=(0.5,1.7), mass_list=[[M_K, M_K],[M_eta, M_pi]], int_N=101)
        >>> p.init_params()
        >>> p2 = DispersionIntegralParticle("A0_980_DI", mass_range=(0.5,1.7), mass_list=[[M_K, M_K],[M_eta, M_pi]], int_N=1001)
        >>> p2.init_params()
        >>> y1 = p.rho_prime(s, M_K, M_K)* (s)
        >>> x1 = p.int_f[0](s) * (s)/np.pi
        >>> x2 = p2.int_f[0](s) * (s)/np.pi
        >>> p_ref = DispersionIntegralParticle("A0_980_DI", mass_range=(0.6,1.6), mass_list=[[M_K, M_K],[M_eta, M_pi]], int_N=11, int_method="scipy")
        >>> p_ref.init_params()
        >>> s_ref = np.linspace(0.6**2, 1.6**2-1e-6, 11)
        >>> x_ref = p_ref.int_f[0](s_ref) * (s_ref)/np.pi
        >>> x_chew = chew_mandelstam(m, M_K, M_K)
        >>> x_qm = 2 * complex_q(s, M_K, M_K).numpy() / m
        >>> _ = plt.plot(m, (x1 - np.max(x2)), label="Re $\\\\Pi (s) - \\\\Pi(s_{th}), N=101$")
        >>> _ = plt.plot(m, (x2 - np.max(x2)), label="Re $\\\\Pi (s) - \\\\Pi(s_{th}), N=1001$")
        >>> _ = plt.plot(m, y1, label="Im $\\\\Pi (s)$")
        >>> _ = plt.plot(m, tf.math.real(x_chew).numpy(), label="$Re\\\\Sigma(s)$", ls="--")
        >>> _ = plt.plot(m, tf.math.imag(x_chew).numpy(), label="$Im \\\\Sigma(s)$", ls="--")
        >>> _ = plt.plot(m, np.real(x_qm), label="2q/m", ls=":")
        >>> _ = plt.scatter(np.sqrt(s_ref), (x_ref - np.max(x2)), label="scipy integration")
        >>> _ = plt.legend()

    The Argand plot

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> M_K = 0.493677
        >>> M_eta = 0.547862
        >>> M_pi = 0.1349768
        >>> from tf_pwa.utils import plot_particle_model
        >>> axis = plot_particle_model("DI", dict(mass=0.98, mass_range=(0.5,1.7), mass_list=[[M_K, M_K],[M_eta, M_pi]], l_list=[0,0], int_N=101), {"R_BC_g_0": 0.415,"R_BC_g_1": 0.405}, mrange=[0.93, 1.05])
        >>> _ = plot_particle_model("DI", dict(mass=0.98, mass_range=(0.5,1.7), mass_list=[[M_K, M_K],[M_eta, M_pi]], l_list=[0,0], int_N=1001), {"R_BC_g_0": 0.415,"R_BC_g_1": 0.405}, mrange=[0.93, 1.05], axis=axis)
        >>> _ = axis[3].legend(["N=101", "N=1001"])


    """

    def __init__(
        self,
        *args,
        mass_range=(0, 5),
        mass_list=[[0.493677, 0.493677]],
        l_list=None,
        int_N=1001,
        int_method="tf",
        dyn_int=False,
        s0=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mass_range = mass_range
        self.srange = (mass_range[0] ** 2, mass_range[1] ** 2)
        self.mass_list = mass_list
        self.int_N = int_N
        self.int_method = int_method
        if l_list is None:
            l_list = [0] * len(mass_list)
        self.l_list = l_list
        self.dyn_int = dyn_int
        if not isinstance(s0, (list, tuple)):
            s0 = [s0] * len(mass_list)
        self.s0 = s0

    def init_params(self):
        super().init_params()
        self.alpha = 2.0
        self.gi = []
        for idx, _ in enumerate(self.mass_list):
            name = f"g_{idx}"
            self.gi.append(self.add_var(name))
        self.init_integral()

    def init_integral(self):
        self.int_f = []
        for idx, ((m1, m2), l, sA) in enumerate(
            zip(self.mass_list, self.l_list, self.s0)
        ):
            fi = lambda s: self.rho_prime(s, m1, m2, l, sA)
            int_fi = DispersionIntegralFunction(
                fi,
                self.srange,
                (m1 + m2) ** 2,
                N=self.int_N,
                method=self.int_method,
            )
            self.int_f.append(int_fi)

    def q2_ch(self, s, m1, m2):
        return (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2) / s / 4

    def rho_prime(self, s, m1, m2, l=0, s0=0.0):
        q2 = self.q2_ch(s, m1, m2)
        q2 = tf.where(s > (m1 + m2) ** 2, q2, tf.zeros_like(q2))
        rho = 2 * tf.sqrt(q2 / s)
        from tf_pwa.breit_wigner import Bprime_q2

        rhop = rho * q2**l * Bprime_q2(l, q2, 1 / self.d**2, self.d) ** 2
        return rhop / self.im_weight(s, m1, m2, l, s0)

    def im_weight(self, s, m1, m2, l=0, s0=0.0):
        if s0 is None or s0 == "sth":
            s0 = (m1 + m2) ** 2
        return s - s0

    def __call__(self, m):
        if self.dyn_int:
            self.init_integral()
        s = m * m
        m0 = self.get_mass()
        s0 = m0 * m0
        gi = tf.stack([var() ** 2 for var in self.gi], axis=-1)
        ims = []
        res = []
        for (m1, m2), li, f, sA in zip(
            self.mass_list, self.l_list, self.int_f, self.s0
        ):
            w = self.im_weight(s, m1, m2, li, sA)
            w0 = self.im_weight(s0, m1, m2, li, sA)
            tmp_i = self.rho_prime(s, m1, m2, li, sA) * w
            tmp_r = f(s) * w - f(s0) * w0
            ims.append(tmp_i)
            res.append(tmp_r)
        im = tf.stack(ims, axis=-1)
        re = tf.stack(res, axis=-1)
        real = s0 - s - tf.reduce_sum(gi * re, axis=-1) / np.pi
        imag = tf.reduce_sum(gi * im, axis=-1)
        dom = real**2 + imag**2
        return tf.complex(real / dom, imag / dom)

    def get_amp(self, *args, **kwargs):
        return self(args[0]["m"])


@register_particle("DI_a0")
class DispersionIntegralParticleA0(DispersionIntegralParticle):
    """

    "DI_a0"  model is the model used in `PRD78,074023(2008) <https://inspirehep.net/literature/793474>`_ . In the model a linear interpolation is used to avoid integration every times in  fitting. No paramters are allowed in the integration, unless `dyn_int=True`.

    .. math::
        f(s) = \\frac{1}{m_0^2 - s - \\sum_{i} [Re \\Pi_i(s) - Re\\Pi_i(m_0^2)] - i \\sum_{i} \\rho'_i(s) }

    where :math:`\\rho'_i(s) = g_i^2 \\rho_i(s) F_i^2(s)` is the phase space with barrier factor :math:`F_i(s)=\\exp(-\\alpha k_i^2)`.

    The real parts of :math:`\\Pi(s)` is defined using the dispersion intergral

    .. math::

        Re \\Pi_i(s) = \\frac{1}{\\pi} P \\int_{s_{th,i}}^{\\infty} \\frac{\\rho'_i(s')}{s' - s} \\mathrm{d} s' = \\lim_{\\epsilon \\rightarrow 0} \\left[ \\int_{s_{th,i}}^{s-\\epsilon} \\frac{\\rho'_i(s')}{s' - s} \\mathrm{d} s' +\\int_{s+\\epsilon}^{\\infty} \\frac{\\rho'_i(s')}{s' - s} \\mathrm{d} s'\\right]

    The reprodution of the Fig1 in  `PRD78,074023(2008) <https://inspirehep.net/literature/793474>`_ .

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> from tf_pwa.amp.dispersion_integral import DispersionIntegralParticleA0
        >>> plt.clf()
        >>> m = np.linspace(0.6, 1.6, 1001)
        >>> s = m * m
        >>> M_K = 0.493677
        >>> M_eta = 0.547862
        >>> M_pi = 0.1349768
        >>> p = DispersionIntegralParticleA0("A0_980_DI", mass_range=(0,2.0), mass_list=[[M_K, M_K],[M_eta, M_pi]], int_N=101)
        >>> p.init_params()
        >>> y1 = p.rho_prime(s, *p.mass_list[0])
        >>> scale1 = 1/np.max(y1)
        >>> x1 = p.int_f[0](s)/np.pi
        >>> p.alpha = 2.5
        >>> p.init_integral()
        >>> y2 = p.rho_prime(s, *p.mass_list[0])
        >>> scale2 = 1/np.max(y2)
        >>> x2 = p.int_f[0](s)/np.pi
        >>> p_ref = DispersionIntegralParticleA0("A0_980_DI", mass_range=(0.6,1.6), mass_list=[[M_K, M_K],[M_eta, M_pi]], int_N=11, int_method="scipy")
        >>> p_ref.init_params()
        >>> s_ref = np.linspace(0.6**2, 1.6**2-1e-6, 11)
        >>> x_ref = p_ref.int_f[0](s_ref)/np.pi
        >>> _ = plt.plot(m, y1* scale1, label="$\\\\rho'(s)$")
        >>> _ = plt.plot(m, x1* scale1, label="Re $\\\\Pi (s)$")
        >>> _ = plt.plot(m, y2* scale2, linestyle="--", label="$\\\\rho'(s),\\\\alpha=2.5$")
        >>> _ = plt.plot(m, x2* scale2, linestyle="--", label="Re $\\\\Pi (s),\\\\alpha=2.5$")
        >>> _ = plt.scatter(np.sqrt(s_ref), x_ref* scale1, label="scipy integration")
        >>> _ = plt.legend()

    The Argand plot

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> M_K = 0.493677
        >>> M_eta = 0.547862
        >>> M_pi = 0.1349768
        >>> from tf_pwa.utils import plot_particle_model
        >>> _ = plot_particle_model("DI_a0", dict(mass=0.98, mass_range=(0,2.0), mass_list=[[M_K, M_K],[M_eta, M_pi]], int_N=101), {"R_BC_g_0": 0.415,"R_BC_g_1": 0.405}, mrange=[0.93, 1.05])

    """

    def rho_prime(self, s, m1, m2, l=0, s0=None):
        q2 = self.q2_ch(s, m1, m2)
        q2 = tf.where(s > (m1 + m2) ** 2, q2, tf.zeros_like(q2))
        rho = 2 * tf.sqrt(q2 / s)
        F = tf.exp(-self.alpha * q2)
        return rho * F**2 / self.im_weight(s, m1, m2, l, s0)

    def im_weight(self, s, m1, m2, l=0, s0=None):
        return tf.ones_like(s)
