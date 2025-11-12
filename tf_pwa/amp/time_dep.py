import math

import tensorflow as tf

from tf_pwa.amp.amp import (
    BaseAmplitudeModel,
    create_amplitude,
    register_amp_model,
)
from tf_pwa.amp.core import HelicityDecay, register_decay, to_complex
from tf_pwa.config import get_config
from tf_pwa.data import data_shape


def cal_gp_gm(t, gamma, delta_m, delta_gamma):
    r"""
    .. math::
        g_{+}(t) = \left[\cosh\frac{\Delta\Gamma t}{4}\cos\frac{\Delta mt}{2}-i\sinh\frac{\Delta\Gamma t}{4}\sin\frac{\Delta mt}{2}\right]\exp\left[-\frac{\Gamma t}{2}\right],

    .. math::
        g_{-}(t) = \left[-\sinh\frac{\Delta\Gamma t}{4}\cos\frac{\Delta mt}{2}+i\cosh\frac{\Delta\Gamma t}{4}\sin\frac{\Delta mt}{2}\right]\exp\left[-\frac{\Gamma t}{2}\right].

    :math:`\exp\left[-i mt\right]` is not included, since its has :math:`|A|=1`.

    """

    decay = tf.exp(-t / 2 * gamma)
    dmt = delta_m * t / 2
    cmt = tf.math.cos(dmt)
    smt = tf.math.sin(dmt)
    dgt = delta_gamma * t / 4
    cgt = tf.math.cosh(dgt)
    sgt = tf.math.sinh(dgt)
    gp = tf.complex(decay * cgt * cmt, -decay * sgt * smt)
    gm = tf.complex(-decay * sgt * cmt, decay * cgt * smt)
    return gp, gm


@register_decay("time_dep_params")
class TimeDepParamsHelicityDecay(HelicityDecay):
    """
    model with `g_ls` and `g_lsb` which dependent on the `tag` in data.

    """

    def init_params(self):
        super().init_params()
        ls = self.get_ls_list()
        self.g_ls_bar = self.add_var(
            "g_lsb", is_complex=True, polar=self.params_polar, shape=(len(ls),)
        )

    def get_mix_g_ls(self, data, data_p, **kwargs):
        g_ls = self.get_g_ls()
        g_ls_bar = self.get_g_ls_bar()
        all_data = kwargs["all_data"]
        ones = tf.ones_like(data_p[self.core]["m"])
        tag = all_data.get("tag", ones)
        return tf.where(tag[..., None] > 0, g_ls, g_ls_bar)

    def get_ls_amp(self, data, data_p, **kwargs):
        g_ls = self.get_mix_g_ls(data, data_p, **kwargs)
        q0 = self.get_relative_momentum2(data_p, False)
        data["|q0|2"] = q0
        q = self.cache_relative_p2(data, data_p)
        if self.has_barrier_factor:
            bf = self.get_barrier_factor2(
                data_p[self.core]["m"], q, q0, self.d
            )
            mag = g_ls
            bf = to_complex(bf)
            m_dep = mag * tf.cast(bf, mag.dtype)
        else:
            m_dep = g_ls  # tf.reshape(g_ls, (1, -1))
        return m_dep

    def get_g_ls_bar(self):
        gls = self.g_ls_bar()
        if self.ls_index is None:
            ret = tf.stack(gls)
        else:
            ret = tf.stack([gls[k] for k in self.ls_index])
        if self.mask_factor:
            return tf.ones_like(ret)
        return ret


@register_decay("time_dep_a")
class TimeDepAHelicityDecay(HelicityDecay):
    """
    model with `g_ls` which dependent on the `tag` in data.

    .. math::
        \\delta_{tag,1}\\sqrt{1-A_{p}} g_{+}(t)  g_{ls} + \\delta_{tag,-1} \\sqrt{1+A_{p}}\\frac{p}{q} g_{-}(t) g_{ls}

    """

    def init_params(self, *args, **kwargs):
        super().init_params(*args, **kwargs)
        self.core.delta_m = self.core.add_var("delta_m", value=0.0)
        self.core.delta_gamma = self.core.add_var("delta_gamma", value=0.0)
        self.core.gamma = self.core.add_var("gamma", value=1.0)
        self.core.poq = self.core.add_var(
            "poq", is_complex=True, fix=True, fix_vals=(1.0, 0.0)
        )
        self.core.A_prod = self.core.add_var(
            "A_prod", range_=(-1, 1), value=0.0
        )

    def get_ls_amp(self, data, data_p, **kwargs):
        m_dep = super().get_ls_amp(data, data_p, **kwargs)
        all_data = kwargs["all_data"]
        ones = tf.ones_like(data_p[self.core]["m"])

        t = all_data.get("time", 0.0 * ones)
        gp, gm = cal_gp_gm(
            t,
            self.core.gamma(),
            self.core.delta_m(),
            self.core.delta_gamma(),
        )
        phase = self.core.poq()
        prod1 = tf.cast(tf.sqrt(1 - self.core.A_prod()), phase.dtype)
        prod2 = tf.cast(tf.sqrt(1 + self.core.A_prod()), phase.dtype)
        ret1 = gp[..., None] * m_dep * prod1
        ret2 = 1 / phase * gm[..., None] * m_dep * prod2
        tag = all_data.get("tag", ones)
        ret = tf.where(tag[..., None] > 0, ret1, ret2)
        return ret


@register_decay("time_dep_abar")
class TimeDepAbarHelicityDecay(HelicityDecay):
    """
    model with `g_ls` (which is `g_lsb` in other model) which dependent on the `tag` in data.

    .. math::
        \\delta_{tag,1}\\sqrt{1-A_{p}} \\frac{q}{p} g_{-}(t)  g_{ls} + \\delta_{tag,-1} \\sqrt{1+A_{p}} g_{+}(t) g_{ls}

    """

    def init_params(self, *args, **kwargs):
        super().init_params(*args, **kwargs)
        self.core.delta_m = self.core.add_var("delta_m", value=0.0)
        self.core.delta_gamma = self.core.add_var("delta_gamma", value=0.0)
        self.core.gamma = self.core.add_var("gamma", value=1.0)
        self.core.poq = self.core.add_var(
            "poq", is_complex=True, fix=True, fix_vals=(1.0, 0.0)
        )
        self.core.A_prod = self.core.add_var(
            "A_prod", range_=(-1, 1), value=0.0
        )

    def get_ls_amp(self, data, data_p, **kwargs):
        m_dep = super().get_ls_amp(data, data_p, **kwargs)
        all_data = kwargs["all_data"]
        ones = tf.ones_like(data_p[self.core]["m"])

        t = all_data.get("time", 0.0 * ones)
        gp, gm = cal_gp_gm(
            t,
            self.core.gamma(),
            self.core.delta_m(),
            self.core.delta_gamma(),
        )
        phase = self.core.poq()
        prod1 = tf.cast(tf.sqrt(1 - self.core.A_prod()), phase.dtype)
        prod2 = tf.cast(tf.sqrt(1 + self.core.A_prod()), phase.dtype)
        ret1 = phase * gm[..., None] * m_dep * prod1
        ret2 = gp[..., None] * m_dep * prod2
        tag = all_data.get("tag", ones)
        ret = tf.where(tag[..., None] > 0, ret1, ret2)
        return ret


@register_decay("time_dep_gls")
class TimeDepHelicityDecay(TimeDepParamsHelicityDecay):
    r"""
    Implement time effect  `arxiv:0904.1869 <https://arxiv.org/abs/0904.1869>`_

    .. math::
        |M(t)\rangle=g_{+}(t)|M\rangle + \frac{q}{p} g_{-}(t)|\bar{M}\rangle, |\bar{M}(t)\rangle=\frac{p}{q}g_{-}(t)|M\rangle + g_{+}(t)|\bar{M}\rangle,

    :math:`|M\rangle` will use `g_ls` and :math:`|\bar{M}\rangle` will use `g_lsb`.

    A factor :math:`\sqrt{1\\pm A_{p}}` is add to include production asymmetry.

    """

    def init_params(self, *args, **kwargs):
        super().init_params(*args, **kwargs)
        self.core.delta_m = self.core.add_var("delta_m", value=0.0)
        self.core.delta_gamma = self.core.add_var("delta_gamma", value=0.0)
        self.core.gamma = self.core.add_var("gamma", value=1.0)
        self.core.poq = self.core.add_var(
            "poq", is_complex=True, fix=True, fix_vals=(1.0, 0.0)
        )
        self.core.A_prod = self.core.add_var(
            "A_prod", range_=(-1, 1), value=0.0
        )

    def get_mix_g_ls(self, data, data_p, **kwargs):
        g_ls = self.get_g_ls()
        g_ls_bar = self.get_g_ls_bar()
        all_data = kwargs["all_data"]
        ones = tf.ones_like(data_p[self.core]["m"])

        t = all_data.get("time", 0.0 * ones)
        gp, gm = cal_gp_gm(
            t,
            self.core.gamma(),
            self.core.delta_m(),
            self.core.delta_gamma(),
        )
        phase = self.core.poq()
        prod1 = tf.cast(tf.sqrt(1 - self.core.A_prod()), dtype=phase.dtype)
        prod2 = tf.cast(tf.sqrt(1 + self.core.A_prod()), dtype=phase.dtype)
        ret1 = gp[..., None] * g_ls + phase * gm[..., None] * g_ls_bar
        ret2 = 1 / phase * gm[..., None] * g_ls + gp[..., None] * g_ls_bar
        tag = all_data.get("tag", ones)
        ret = tf.where(tag[..., None] > 0, prod1 * ret1, prod2 * ret2)
        return ret


@register_amp_model("time_dep_params")
class TimeDepParamsAmplitudeModel(BaseAmplitudeModel):
    """
    Implement time effect `arxiv:0904.1869 <https://arxiv.org/abs/0904.1869>`_
    Require to use decay model `time_dep_params`.

    """

    def init_params(self, *args, **kwargs):
        super().init_params()
        top = self.decay_group.top
        top.delta_m = top.add_var("delta_m", value=0.0)
        top.delta_gamma = top.add_var("delta_gamma", value=0.0)
        top.gamma = top.add_var("gamma", value=1.0)
        top.poq = top.add_var(
            "poq", is_complex=True, fix=True, fix_vals=(1.0, 0.0), polar=True
        )
        top.A_prod = top.add_var("A_prod", range_=(-1, 1), value=0.0)

    def eval_A_Abar(self, data):
        top = self.decay_group.top
        ones = tf.ones((1,), dtype=get_config("dtype"))
        A = self.decay_group.get_amp2({**data, "tag": ones})
        Abar = self.decay_group.get_amp2({**data, "tag": -ones})
        return A, Abar

    def eval_A_Abar_time(self, data):
        A, Abar = self.eval_A_Abar(data)
        top = self.decay_group.top
        ones = tf.ones((1,), dtype=get_config("dtype"))
        t = data.get("time", 0.0 * ones)
        gp, gm = cal_gp_gm(t, top.gamma(), top.delta_m(), top.delta_gamma())
        n_pad = len(A.shape) - len(t.shape)
        phase = top.poq()
        shape = A.shape
        size = 1
        for i in range(n_pad):
            size *= shape[-i - 1]
        gp = tf.reshape(gp, (-1, 1))
        gm = tf.reshape(gm, (-1, 1))
        A = tf.reshape(A, (-1, size))
        Abar = tf.reshape(Abar, (-1, size))
        ret1 = gp * A + phase * gm * Abar
        ret2 = 1 / phase * gm * A + gp * Abar
        if shape[0] is None:
            shape = (-1, *shape[1:])
        return tf.reshape(ret1, shape), tf.reshape(ret2, shape)

    def pdf(self, data):
        A, Abar = self.eval_A_Abar_time(data)
        ret1 = self.decay_group.sum_with_polarization(A)
        ret2 = self.decay_group.sum_with_polarization(Abar)
        ones = tf.ones((1,), dtype=get_config("dtype"))
        tag = data.get("tag", ones)
        top = self.decay_group.top
        prod1 = tf.cast((1 - top.A_prod()), dtype=ret1.dtype)
        prod2 = tf.cast((1 + top.A_prod()), dtype=ret2.dtype)
        return tf.where(tag > 0, ret1 * prod1, ret2 * prod2)


@register_amp_model("time_dep_params_fs")
class TimeDepParamsFSAmplitudeModel(TimeDepParamsAmplitudeModel):
    """

    Flavour specific version of `time_dep_params`.

    """

    def eval_A2_Abar2_time(self, data):
        A, Abar = self.eval_A_Abar(data)

        top = self.decay_group.top
        ones = tf.ones((1,), dtype=get_config("dtype"))
        t = data.get("time", 0.0 * ones)
        gp, gm = cal_gp_gm(t, top.gamma(), top.delta_m(), top.delta_gamma())
        n_pad = len(A.shape) - len(t.shape)
        phase = top.poq()
        for i in range(n_pad):
            gp = tf.expand_dims(gp, -1)
            gm = tf.expand_dims(gm, -1)
        ret1_a = self.decay_group.sum_with_polarization(gp * A)
        ret1_abar = self.decay_group.sum_with_polarization(phase * gm * Abar)
        ret2_a = self.decay_group.sum_with_polarization(1 / phase * gm * A)
        ret2_abar = self.decay_group.sum_with_polarization(gp * Abar)
        ret1 = ret1_a + ret1_abar
        ret2 = ret2_a + ret2_abar
        return ret1, ret2

    def pdf(self, data):
        ret1, ret2 = self.eval_A2_Abar2_time(data)
        ones = tf.ones((1,), dtype=get_config("dtype"))
        tag = data.get("tag", ones)
        top = self.decay_group.top
        prod1 = tf.cast((1 - top.A_prod()), dtype=ret1.dtype)
        prod2 = tf.cast((1 + top.A_prod()), dtype=ret2.dtype)
        return tf.where(tag > 0, ret1 * prod1, ret2 * prod2)


@register_amp_model("time_dep_cp")
class TimeDepCpAmplitudeModel(TimeDepParamsAmplitudeModel):
    """
    Time dependent amplitude with self-CP related process, the Abar will calculate through :math:`\\bar{A}(p_{+}, p_{-}, p_{0}) = A(-p_{-}, -p_{+}, -p_{0})`
    """

    def init_params(self, *args, **kwargs):
        super().init_params(*args, **kwargs)
        top = self.decay_group.top
        top.poq.freed()

    def eval_A_Abar(self, data):
        one = tf.ones_like(data["time"])
        A = self.decay_group.get_amp2({**data, "tag": one})
        Abar = self.decay_group.get_amp2({**data["cp_swap"], "tag": -one})
        return A, Abar


@register_amp_model("time_dep_cp_fs")
class TimeDepCpFSAmplitudeModel(TimeDepParamsFSAmplitudeModel):
    """
    Flavour specific version of `time_dep_cp`.
    """

    def init_params(self, *args, **kwargs):
        super().init_params(*args, **kwargs)
        top = self.decay_group.top
        top.poq.freed()

    def eval_A_Abar(self, data):
        one = tf.ones_like(data["time"])
        A = self.decay_group.get_amp2({**data, "tag": one})
        Abar = self.decay_group.get_amp2({**data["cp_swap"], "tag": -one})
        return A, Abar


@register_amp_model("time_dep_params_conv")
class TimeDepParamsConvAmplitudeModel(TimeDepParamsAmplitudeModel):
    """
    .. math::
        |A(t)| = \\int | A(\\tau)|^2 \\frac{1}{\\sqrt{2\\pi}\\sigma} \\exp(-\\frac{(t-\\tau)^2}{2\\sigma^2}) d\\tau

    Convolve with a Gaussian function

    """

    def __init__(self, *args, t_min=0.0, **kwargs):
        self.t_min = t_min
        super().__init__(*args, **kwargs)

    def pdf(self, data):
        A, Abar = self.eval_A_Abar(data)
        top = self.decay_group.top
        phase = top.poq()

        ones = tf.ones((1,), dtype=get_config("dtype"))
        t = data.get("time", 0.0 * ones)
        sigma = data.get("time_sigma", 0.0 * ones)
        # -(Gamma - Delta\Gamma/2)
        exp_pgamma_t = conv_exp_gaussian(
            t, sigma, top.gamma() - top.delta_gamma() / 2, self.t_min
        )
        # -(Gamma + Delta\Gamma/2)
        exp_mgamma_t = conv_exp_gaussian(
            t, sigma, top.gamma() + top.delta_gamma() / 2, self.t_min
        )
        # -(Gamma + Delta\Gamma/2)
        exp_dm_t = conv_exp_gaussian_complex(
            t, sigma, top.gamma(), -top.delta_m(), self.t_min
        )

        cosht = (exp_pgamma_t + exp_mgamma_t) / 2
        sinht = (exp_pgamma_t - exp_mgamma_t) / 2
        cost = tf.math.real(exp_dm_t)
        sint = tf.math.imag(exp_dm_t)
        # print(cost, sint, cosht, sinht)

        A2 = self.decay_group.sum_with_polarization(A)
        Abar2 = self.decay_group.sum_with_polarization(phase * Abar)
        Asum = A2 + Abar2
        Asub = A2 - Abar2
        ReA = self.decay_group.sum_with_polarization(phase * Abar, A)
        ImA = self.decay_group.sum_with_polarization(
            phase * Abar, 1j * A
        )  # conj(jA)=-jA

        ret1 = (
            Asum * cosht + Asub * cost - 2 * ReA * sinht - 2 * ImA * sint
        ) / 2

        # A <-> Abar ,  q/p <-> p/q = 1/( q/p )
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

        tag = data.get("tag", ones)
        prod1 = tf.cast((1 - top.A_prod()), dtype=ret1.dtype)
        prod2 = tf.cast((1 + top.A_prod()), dtype=ret2.dtype)
        ret = tf.where(tag > 0, ret1 * prod1, ret2 * prod2)
        return ret


@register_amp_model("time_dep_cp_conv")
class TimeDepCpConvAmplitudeModel(TimeDepParamsConvAmplitudeModel):
    """
    Time dependent amplitude with self-CP related process, the Abar will calculate through :math:`\\bar{A}(p_{+}, p_{-}, p_{0}) = A(-p_{-}, -p_{+}, -p_{0})`
    """

    def init_params(self, *args, **kwargs):
        super().init_params(*args, **kwargs)
        top = self.decay_group.top
        top.poq.freed()

    def eval_A_Abar(self, data):
        A = self.decay_group.get_amp2(data)
        Abar = self.decay_group.get_amp2(data["cp_swap"])
        return A, Abar


@register_amp_model("flavour_tag")
class FlavourTagPDF(BaseAmplitudeModel):
    def __init__(
        self,
        *args,
        eta_name="eta",
        tag_name="tag_value",
        true_tag="tag",
        tag_eff=1.0,
        **kwargs,
    ):
        self.eta_name = eta_name
        self.tag_name = tag_name
        self.true_tag = true_tag
        self.tag_eff = tag_eff
        super().__init__(*args, **kwargs)

    def pdf(self, data):
        eta = data.get(self.eta_name, 0.0)
        tag_value = data.get(self.tag_name, 1)
        true_tag = data.get(self.true_tag, 1)
        tag1, tag2 = eta, eta
        return tf.where(
            true_tag > 0,
            tf.where(
                tag_value == 0,
                tf.ones_like(tag1) * (1 - self.tag_eff),
                tf.where(tag_value > 0, 1 - tag1, tag1) * self.tag_eff,
            ),
            tf.where(
                tag_value == 0,
                tf.ones_like(tag2) * (1 - self.tag_eff),
                tf.where(tag_value > 0, tag2, 1 - tag2) * self.tag_eff,
            ),
        )


@register_amp_model("flavour_tag_linear")
class FlavourTagLinearPDF(BaseAmplitudeModel):
    def __init__(
        self,
        *args,
        eta_name="eta",
        eta_mean=[0.5, 0.5],
        tag_name="tag_value",
        true_tag="tag",
        tag_eff=[1.0, 1.0],
        prefix="",
        **kwargs,
    ):
        self.eta_name = eta_name
        self.tag_name = tag_name
        self.true_tag = true_tag
        self.eta_mean = eta_mean
        self.tag_eff = tag_eff
        self.prefix = prefix
        super().__init__(*args, **kwargs)

    def init_params(self, *args, **kwargs):
        super().init_params(*args, **kwargs)
        self.top = self.decay_group.top
        self.top.p0 = self.top.add_var(
            self.prefix + "p0", value=self.eta_mean[0]
        )
        self.top.p1 = self.top.add_var(self.prefix + "p1", value=1.0)
        self.top.p0bar = self.top.add_var(
            self.prefix + "p0bar", value=self.eta_mean[1]
        )
        self.top.p1bar = self.top.add_var(self.prefix + "p1bar", value=1.0)

    def pdf(self, data):
        eta = data.get(self.eta_name, 0.0)
        true_tag = data.get(self.true_tag, 1)
        tag_value = data.get(self.tag_name, true_tag)
        tag1 = self.top.p0() + self.top.p1() * (eta - self.eta_mean[0])
        tag2 = self.top.p0bar() + self.top.p1bar() * (eta - self.eta_mean[1])
        return tf.where(
            true_tag > 0,
            tf.where(
                tag_value == 0,
                tf.ones_like(tag1) * (1 - self.tag_eff[0]),
                tf.where(tag_value > 0, 1 - tag1, tag1) * self.tag_eff[0],
            ),
            tf.where(
                tag_value == 0,
                tf.ones_like(tag2) * (1 - self.tag_eff[1]),
                tf.where(tag_value > 0, tag2, 1 - tag2) * self.tag_eff[1],
            ),
        )


@register_amp_model("flavour_tag_mix")
class TimeDepFTPDF(BaseAmplitudeModel):
    def __init__(
        self,
        decay_group,
        base_model={"model": "default"},
        taggers=[{"model": "flavour_tag"}],
        **kwargs,
    ):
        self.base_model = create_amplitude(decay_group, **base_model)
        self.taggers = [create_amplitude(decay_group, **i) for i in taggers]
        super().__init__(decay_group, **kwargs)

    def init_params(self, *args, **kwargs):
        super().init_params(*args, **kwargs)
        self.base_model.init_params(*args, **kwargs)
        for tagger in self.taggers:
            tagger.init_params(*args, **kwargs)

    def pdf(self, data):
        time = data["time"]
        tag = tf.ones_like(time)
        old_tag = data.get("tag", None)
        data["tag"] = tag
        amp1 = self.base_model(data)
        ft1 = tf.reduce_prod([f(data) for f in self.taggers], axis=0)
        data["tag"] = -tag
        amp2 = self.base_model(data)
        ft2 = tf.reduce_prod([f(data) for f in self.taggers], axis=0)
        del data["tag"]
        if old_tag is not None:
            data["tag"] = old_tag
        return amp1 * ft1 + amp2 * ft2


def fix_cp_params(config, r1, r2):
    """
    using the same paramters for A and Abar

    only work for dalitz plot (all J=0 for initial and final particles)

    """
    config.get_params()  # asume parameters exist
    decay_group = config.get_decay()
    fix_params = {}
    same_var = []
    free_var = []
    for a, b in zip(r1, r2):
        chain_a = decay_group.get_decay_chain(a)
        chain_b = decay_group.get_decay_chain(b)
        dec1 = [i for i in chain_a if i.core == chain_a.top][0]
        dec2 = [i for i in chain_b if i.core == chain_b.top][0]
        dec1.g_ls.sameas(dec2.g_ls_bar)
        dec2.g_ls.sameas(dec1.g_ls_bar)
        if not (chain_a.total.is_fixed()):
            dec1.g_ls.set_fix_idx(free_idx=0)
        if not (chain_b.total.is_fixed()):
            dec2.g_ls.set_fix_idx(free_idx=0)
        chain_a.total.sameas(chain_b.total)
        chain_a.total.set_fix_idx(0, fix_vals=1.0)
    for i in decay_group.resonances:
        if str(i) not in r1 and str(i) not in r2:
            chain = decay_group.get_decay_chain(i)
            dec = [i for i in chain if i.core == chain.top][0]
            if i.J % 2 == 1:  # (-1)^l
                dec.g_ls_bar.set_fix_idx(0, fix_vals=-1.0)
            else:
                dec.g_ls_bar.set_fix_idx(0, fix_vals=1.0)


def fix_cp_params_aabar(config, r1, r2):
    """
    using the same paramters for A and Abar of `time_dep_a` and `time_dep_abar`

    only work for dalitz plot (all J=0 for initial and final particles)

    """
    config.get_params()  # asume parameters exist
    decay_group = config.get_decay()
    for a, b in zip(r1, r2):
        chain_a = decay_group.get_decay_chain(a)
        chain_b = decay_group.get_decay_chain(b)
        dec1 = [i for i in chain_a if i.core == chain_a.top][0]
        dec2 = [i for i in chain_b if i.core == chain_b.top][0]
        chain_a.total.sameas(chain_b.total)
        for i, j in zip(chain_a, chain_b):
            i.g_ls.sameas(j.g_ls)


@tf.custom_gradient
def erfc_xy(x, y):
    z = tf.complex(x, y)
    from scipy.special import erfc as erfc_scipy

    e = tf.numpy_function(erfc_scipy, (z,), Tout=tf.complex128)
    rx, ry = tf.math.real(e), tf.math.imag(e)

    def grad(gx, gy):
        gz = -2 / math.sqrt(math.pi) * tf.exp(-z * z)
        gzx, gzy = tf.math.real(gz), tf.math.imag(gz)
        return gx * gzx + gy * gzy, gy * gzx - gx * gzy

    return (rx, ry), grad


def erfc(z):
    x, y = tf.math.real(z), tf.math.imag(z)
    rx, ry = erfc_xy(x, y)
    return tf.complex(rx, ry)


def conv_exp_gaussian(t, sigma, a, left=0.0):
    """

    .. math::
        \\int_{l}^{\\infty} \\exp(-a\\tau) \\frac{1}{\\sqrt{2\\pi}\\sigma} \\exp(-\\frac{(t - \\tau)^2}{2\\sigma^2}) d \\tau
        = \\frac{\\exp[- ax + \\frac{a^2\\sigma^2}{2}]}{2}\\text{erfc}\\frac{a\\sigma^2 + l - x}{\\sqrt{2}\\sigma}

    """
    s2 = math.sqrt(2)
    z = a * sigma / s2 - (t - left) / sigma / s2
    sigma_sq = sigma * sigma
    e = a * a / 2 * sigma_sq - a * t
    return tf.exp(e) * tf.math.erfc(z) / 2


def conv_exp_gaussian_complex(t, sigma, a, b, left=0.0):
    """

    .. math::
        \\int_{l}^{\\infty} \\exp(-(a+bi) \\tau) \\frac{1}{\\sqrt{2\\pi}\\sigma} \\exp(-\\frac{(t - \\tau)^2}{2\\sigma^2}) d \\tau
        = \\frac{\\exp[- (a+bi)x + \\frac{(a+bi)^2 \\sigma^2}{2}]}{2}\\text{erfc}\\frac{(a+bi)\\sigma^2 + l - x}{\\sqrt{2}\\sigma}
        = \\exp(-\\frac{x^2}{2s^2}) \\text{Faddeeva}(i\\frac{(a+bi)\\sigma^2 + l - x}{\\sqrt{2}\\sigma} )

    """
    s2 = math.sqrt(2)
    z = tf.complex(a * sigma / s2 - (t - left) / sigma / s2, b * sigma / s2)
    sigma_sq = sigma * sigma
    e = tf.complex(
        (a * a - b * b) / 2 * sigma_sq - a * t, a * b * sigma_sq - b * t
    )
    return tf.exp(e) * erfc(z) / 2
