import tensorflow as tf

from tf_pwa.amp.core import (
    HelicityDecay,
    Particle,
    get_relative_p2,
    register_decay,
    register_particle,
)
from tf_pwa.breit_wigner import BWR2, Bprime_q2, to_complex


@register_decay("LS-decay")
class ParticleDecayLS(HelicityDecay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.same_phase = kwargs.get("same_phase", False)
        self.same_ratio = kwargs.get("same_ratio", False)

    def init_params(self):
        self.d = 3.0
        ls = self.get_ls_list()
        if len(ls) <= 0:
            print("no aviable ls", self, self.get_ls_list())
            return
        if self.same_ratio:
            if self.same_phase:
                self.g_ls = self.add_var(
                    "g_ls", is_complex=False, shape=(len(ls),)
                )
                for i in range(len(ls)):
                    self.g_ls.set_fix_idx(fix_idx=0, fix_vals=1.0)
            else:
                self.g_ls = self.add_var(
                    "g_ls", is_complex=True, shape=(len(ls),)
                )
                self.g_ls.set_fix_idx(fix_idx=0, fix_vals=(1.0, 0.0))
                self.g_ls.set_same_ratio()
        else:
            if self.same_phase:
                self.g_ls = self.add_var(
                    "g_ls", is_complex=False, shape=(len(ls),)
                )
                self.g_ls.set_fix_idx(fix_idx=0, fix_vals=1.0)
            else:
                self.g_ls = self.add_var(
                    "g_ls", is_complex=True, shape=(len(ls),)
                )
                self.g_ls.set_fix_idx(fix_idx=0, fix_vals=(1.0, 0.0))

    def get_barrier_factor2(self, mass, q2, q02, d):
        ls = self.get_ls_list()
        ls_amp = self.core.get_ls_amp(mass, ls, q2=q2, q02=q02, d=d)
        if self.ls_index is None:
            return tf.stack(ls_amp, axis=-1)
        return tf.stack([ls_amp[k] for k in self.ls_index], axis=-1)


class ParticleLS(Particle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        decay_params = kwargs.get("decay_params", {})
        self.decay_params = {"model": "LS-decay", **decay_params}

    def get_amp(self, *args, **kwargs):
        m = args[0]["m"]
        zeros = tf.zeros_like(m)
        ones = tf.ones_like(m)
        return tf.complex(ones, zeros)

    def get_ls_amp(self, m, ls, q2, q02, d=3):
        raise NotImplementedError

    def is_fixed_shape(self):
        return False


@register_particle("BWR_LS")
class ParticleBWRLS(ParticleLS):
    """

    Breit Wigner with split ls running width

    .. math::
        R_i (m) = \\frac{g_i}{m_0^2 - m^2 - im_0 \\Gamma_0 \\frac{\\rho}{\\rho_0} (\\sum_{i} g_i^2)}

    , :math:`\\rho = 2q/m`, the partial width factor is

    .. math::
        g_i = \\gamma_i \\frac{q^l}{q_0^l} B_{l_i}'(q,q_0,d)

    and keep normalize as

    .. math::
        \\sum_{i} \\gamma_i^2 = 1.

    The normalize is done by (:math:`\\cos \\theta_0, \\sin\\theta_0 \\cos \\theta_1, \\cdots, \\prod_i \\sin\\theta_i`)

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.same_ratio = kwargs.get("same_ratio", True)
        self.same_phase = kwargs.get("same_phase", False)
        self.fix_bug1 = kwargs.get("fix_bug1", False)
        self.decay_params = {
            "same_ratio": self.same_ratio,
            "same_phase": self.same_phase,
            **self.decay_params,
        }
        self.theta = []

    def init_params(self):
        super().init_params()
        if getattr(self, "ls_list", None) is None:
            self.ls_list = self.decay[0].get_ls_list()
        self.theta = []
        for i in range(len(self.ls_list) - 1):
            self.theta.append(self.add_var(f"theta{i}"))

    def factor_gamma(self, ls):
        if len(ls) <= 1:
            return [1.0]
        f = 1.0
        ret = []
        for i in range(len(ls) - 1):
            theta_i = self.theta[i]()
            a = tf.cos(theta_i)
            ret.append(f * a)
            f = f * tf.sin(theta_i)
        ret.append(f)
        return ret

    def get_barrier_factor(self, ls, q2, q02, d):
        return [tf.sqrt(q2 / q02) ** i * Bprime_q2(i, q2, q02, d) for i in ls]

    def get_sympy_var(self):
        import sympy

        m, m0, g0, m1, m2 = sympy.var("m m0 g0 m1 m2")
        theta = sympy.var("theta0:{}".format(len(self.theta)))
        return m, m0, g0, theta, m1, m2

    def get_num_var(self):
        mass = self.get_mass()
        width = self.get_width()
        m1, m2 = self.get_subdecay_mass()
        thetas = [i() for i in self.theta]
        return mass, width, thetas, m1, m2

    def get_sympy_dom(self, m, m0, g0, thetas, m1=None, m2=None):
        if self.get_width() is None:
            raise NotImplemented
        from tf_pwa.formula import BWR_LS_dom

        d = self.decay[0].d if self.decay else 3.0

        return BWR_LS_dom(
            m,
            m0,
            g0,
            thetas,
            self.decay[0].get_l_list(),
            m1,
            m2,
            d=d,
            fix_bug1=self.fix_bug1,
        )

    def __call__(self, m):
        m0 = self.get_mass()
        m1 = self.decay[0].outs[0].get_mass()
        m2 = self.decay[0].outs[1].get_mass()
        ls = self.decay[0].get_ls_list()
        q2 = get_relative_p2(m, m1, m2)
        q02 = get_relative_p2(m0, m1, m2)
        return self.get_ls_amp(m, ls, q2, q02)

    def get_ls_amp(self, m, ls, q2, q02, d=3.0):
        dom, total_gamma = self.get_ls_amp_frac(m, ls, q2, q02, d)

        ret = []
        for i in total_gamma:
            i = to_complex(i)
            ret.append(tf.cast(i, dom.dtype) / dom)

        return ret

    def get_ls_amp_frac(self, m, ls, q2, q02, d=3.0):
        assert all(i in self.ls_list for i in ls)
        ls = [i for i, j in self.ls_list]
        gammai = self.factor_gamma(ls)
        bf = self.get_barrier_factor(ls, q2, q02, d)
        total_gamma = [i * j for i, j in zip(gammai, bf)]
        m0 = self.get_mass()
        g0 = self.get_width()

        a = m0 * m0 - m * m
        b = m0 * g0 * tf.sqrt(q2 / q02) * sum([i * i for i in total_gamma])
        if self.fix_bug1:
            b = b * m0 / m
        else:
            b = b * m / m0
        dom = tf.complex(a, -b)
        return dom, total_gamma


@register_particle("BWR_LS2")
class ParticleBWRLS2(ParticleLS):
    """

    Breit Wigner with split ls running width, each one use their own l,

    .. math::
        R_i (m) = \\frac{1}{m_0^2 - m^2 - im_0 \\Gamma_0 \\frac{\\rho}{\\rho_0} (g_i^2)}

    , :math:`\\rho = 2q/m`, the partial width factor is

    .. math::
        g_i = \\gamma_i \\frac{q^l}{q_0^l} B_{l_i}'(q,q_0,d)

    """

    def __call__(self, m, l=0):
        m0 = self.get_mass()
        m1 = self.decay[0].outs[0].get_mass()
        m2 = self.decay[0].outs[1].get_mass()
        ls = [(l, 0)]
        q2 = get_relative_p2(m, m1, m2)
        q02 = get_relative_p2(m0, m1, m2)
        return self.get_ls_amp(m, ls, q2, q02)

    def get_ls_amp(self, m, ls, q2, q02, d=3.0):
        m0 = self.get_mass()
        g0 = self.get_width()

        d = self.decay[0].d if self.decay else 3.0

        ret = []
        for l, s in ls:
            ret.append(BWR2(m, m0, g0, q2, q02, l, d))
        return ret


@register_particle("MultiBWR")
class ParticleMultiBWR(ParticleLS):
    """

    Combine Multi BWR into one

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> from tf_pwa.utils import plot_particle_model
        >>> axis = plot_particle_model("MultiBWR", {"mass_list": [0.4, 0.6], "width_list":[0.03, 0.04]},
        ... {"R_BC_coeff_0_0r": 0.03, "R_BC_coeff_0_0i": 0.0, "R_BC_coeff_0_1r": 0.04, "R_BC_coeff_0_1i": 0.0})

    """

    def __init__(self, *args, **kwargs):
        kwargs["same_ratio"] = kwargs.get("same_ratio", True)
        kwargs["same_phase"] = kwargs.get("same_phase", True)
        super().__init__(*args, **kwargs)

    def init_params(self):
        if getattr(self, "ls_list", None) is None:
            self.ls_list = self.decay[0].get_ls_list()
        self.all_mass = self.add_var("com_mass", shape=(len(self.mass_list),))
        self.all_mass.set_value(self.mass_list)
        self.all_width = self.add_var(
            "com_width", shape=(len(self.width_list),)
        )
        self.all_width.set_value(self.width_list)
        self.coeff = self.add_var(
            "coeff",
            shape=(len(self.ls_list), len(self.mass_list)),
            is_complex=True,
        )
        self.coeff.set_fix_idx([[0, 0]], [1.0, 0.0])

    def mass(self):
        return self.all_mass()[0]

    def get_barrier_factor(self, ls, q2, q02, d):
        return [
            tf.sqrt(q2 / q02) ** i[0] * Bprime_q2(i[0], q2, q02, d) for i in ls
        ]

    def get_ls_amp(self, m, ls, q2, q02, d=3.0):
        coeff = self.coeff()
        all_mass = self.all_mass()
        all_width = self.all_width()
        l = min([i[0] for i in ls])
        dom = []
        for m0, g0 in zip(all_mass, all_width):
            dom.append(BWR2(m, m0, g0, q2, q02, l, d))
        dom = tf.stack(dom, axis=-1)
        ret = []
        bf = self.get_barrier_factor(ls, q2, q02, d)
        for c, (l, s), bfi in zip(coeff, ls, bf):
            tmp = tf.reduce_sum(dom * tf.stack(c), axis=-1)
            ret.append(tmp * tf.cast(bfi, tmp.dtype))
        return ret
