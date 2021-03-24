import tensorflow as tf

from tf_pwa.amp.core import (
    HelicityDecay,
    Particle,
    register_decay,
    register_particle,
)
from tf_pwa.breit_wigner import Bprime_q2


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
                self.g_ls.set_same_ratio()
                self.g_ls.set_fix_idx(fix_idx=0, fix_vals=(1.0, 0.0))
        else:
            if self.same_phase:
                self.g_ls = self.add_var(
                    "g_ls", is_complex=False, shape=(len(ls),)
                )
                try:
                    self.g_ls.set_fix_idx(fix_idx=0, fix_vals=1.0)
                except Exception as e:
                    print(e, self, self.get_ls_list())
            else:
                self.g_ls = self.add_var(
                    "g_ls", is_complex=True, shape=(len(ls),)
                )
                try:
                    self.g_ls.set_fix_idx(fix_idx=0, fix_vals=(1.0, 0.0))
                except Exception as e:
                    print(e, self, self.get_ls_list())

    def get_barrier_factor2(self, mass, q2, q02, d):
        ls = self.get_l_list()
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


@register_particle("BWR_LS")
class ParticleBWRLS(ParticleLS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decay_params = {"same_ratio": True, **self.decay_params}
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
            return 1.0
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
        return [tf.sqrt(q2 / q02) * Bprime_q2(i, q2, q02, d) for i in ls]

    def get_ls_amp(self, m, ls, q2, q02, d=3.0):
        gammai = self.factor_gamma(ls)
        bf = self.get_barrier_factor(ls, q2, q02, d)
        total_gamma = [i * j for i, j in zip(gammai, bf)]
        m0 = self.get_mass()
        g0 = self.get_width()

        a = m0 * m0 - m * m
        b = (
            m0
            * g0
            * tf.sqrt(q2 / q02)
            * m
            / m0
            * sum([i * i for i in total_gamma])
        )
        dom = tf.complex(a, -b)

        ret = []
        for i in total_gamma:
            ret.append(tf.cast(i, dom.dtype) / dom)

        return ret
