import itertools
import math

import numpy as np
import tensorflow as tf

from tf_pwa.amp import (
    DecayChain,
    HelicityDecay,
    register_decay,
    register_decay_chain,
)
from tf_pwa.angle import LorentzVector as lv
from tf_pwa.data import data_shape
from tf_pwa.particle import BaseDecay, _spin_int


class IndexMap:
    def __init__(self):
        self.used_symbol = []
        self.symbols = {}
        self.all_symbols = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )

    def get(self, name):
        if name not in self.symbols:
            tmp = None
            for i in self.all_symbols:
                if i not in self.used_symbol:
                    tmp = i
                    break
            if tmp is None:
                raise IndexError("all symbol are used")
            self.used_symbol.append(tmp)
            self.symbols[name] = tmp
        return self.symbols[name]


def create_epsilon():
    """
    epsilon^{a}_{bcd}

    """
    from sympy import LeviCivita

    ret = np.zeros((4, 4, 4, 4))
    for i, j, k, l in itertools.permutations([0, 1, 2, 3], 4):
        sign = LeviCivita(i, j, k, l)
        ret[i, j, k, l] = sign * np.array([1, -1, -1, -1])[i]
    return ret


C_epsilon = create_epsilon()


class EinSum:
    def __init__(self, name, idx, inputs=None, replace_axis=[]):
        inputs = {} if inputs is None else inputs
        self.name = name
        self.index = idx
        self.inputs = inputs
        self.replace_axis = dict(replace_axis)
        self.extra_axis = []
        for i in inputs:
            self.extra_axis += i.extra_axis

    def call(self, inputs, cached=None):
        # print(self, inputs)
        cached = {} if cached is None else cached
        real_inputs = []
        for i in self.inputs:
            if i.name == "g":
                real_inputs.append(np.diag([1, -1, -1, -1]))
            elif i.name == "epsilon":
                real_inputs.append(C_epsilon)
            elif i.name in inputs:
                real_inputs.append(inputs[i.name])
            elif i.name in cached:
                real_inputs.append(cached[i.name])
            else:
                assert isinstance(i, EinSum), ""
                real_inputs.append(i.call(inputs, cached))
        idx_map = IndexMap()

        def format_idx(index):
            return "..." + "".join(
                [idx_map.get(self.replace_axis.get(j, j)) for j in index]
            )

        einsum_expr = ",".join(
            [format_idx(list(i.index) + i.extra_axis) for i in self.inputs]
        )
        einsum_expr = (
            einsum_expr + "->" + format_idx(list(self.index) + self.extra_axis)
        )
        # print(self.name, self, self.index, [i.index for i in self.inputs])
        # print(einsum_expr)
        # print(einsum_expr)
        if any(i.dtype is tf.complex128 for i in real_inputs):
            real_inputs = [tf.cast(i, tf.complex128) for i in real_inputs]
        # print(self, real_inputs)
        # print([i.shape for i in real_inputs])
        ret = tf.einsum(einsum_expr, *real_inputs)
        # print(self.name, ret)
        return ret

    def set_inputs(self, name, value):
        inputs = list(self.inputs)
        for i, j in enumerate(inputs):
            if j.name == name:
                self.inputs[i] = value
            j.set_inputs(name, value)

    def insert_extra_axis(self, name, indexs):
        if name == self.name:
            self.extra_axis += indexs
            return indexs
        insert = []
        for i in self.inputs:
            a = i.insert_extra_axis(name, indexs)
            insert += a
        self.extra_axis += insert
        return insert

    def __repr__(self):
        return f"{self.name}[{','.join([str(i) for i in self.inputs])}]"


class EinSumCall(EinSum):
    def __init__(self, name, idx, function):
        super().__init__(name, idx, [])
        self.function = function
        self.extra_axis = []

    def call(self, inputs, cached=None):
        cached = {} if cached is None else cached
        if self.name in cached:
            ret = cached[self.name]
        else:
            ret = self.function(inputs, cached)
        # print(self.name, ret)
        return ret


def mass2(t):
    return tf.reduce_sum(t**2 * np.array([1, -1, -1, -1]), axis=-1)


def dot(p1, p2):
    return tf.reduce_sum(p1 * p2 * np.array([1, -1, -1, -1]), axis=-1)


class EvalT:
    """
    t^{u}

    """

    def __init__(self, decay, l):
        self.decay = decay
        self.l = l

    def __call__(self, inputs, cached=None):
        cached = {} if cached is None else cached
        p1 = inputs[f"{self.decay.outs[0]}_p"]
        p2 = inputs[f"{self.decay.outs[1]}_p"]
        p0 = p1 + p2  # inputs[f"{self.decay.core}_p"]
        # test = EvalP(self.decay.core, self.l)
        if self.l == 0:
            return tf.ones_like(p1[..., 0])
        elif self.l == 1:
            r = p1 - p2  # r^u
            # rt^u = - g^u_v r^v + (p1 - p2)^v (p1+p2)_v (p1+p2)^u
            r_t = (
                -r + ((mass2(p1) - mass2(p2)) / mass2(p0))[..., None] * p0
            )  #  * np.array([1,-1,-1,-1])
            # print( "diff", tf.reduce_sum(test(inputs) * r[...,None,:], axis=-1) - r_t)
            return r_t
        elif self.l == 2:
            r = p1 - p2
            g_t = (
                np.diag([1, -1, -1, -1])
                - p0[..., None] * p0[..., None, :] / mass2(p0)[..., None, None]
            )  # gbar^uv =  g^{uv} - p^u p^v / p^2
            # r_t_a =  -r + ((mass2(p1) - mass2(p2))/mass2(p0))[...,None] * p0
            r_t = tf.reduce_sum(
                g_t * r[..., None, :] * np.array([1, -1, -1, -1]), axis=-1
            )  # rt^u = gbar^uv g_vo r^o
            # t^uv =  rt^u rt^v + 1/3 r_t^2 gbar^uv
            ret = (
                r_t[..., None, :] * r_t[..., None]
                - 1 / 3 * dot(r_t, r_t)[..., None, None] * g_t
            )
            # print( "diff", tf.reduce_sum(test(inputs) * r[...,None,None,None,:] * r[...,None, None, :,None], axis=[-1,-2]) - ret)
            b = EvalBoost([self.decay.outs[0], self.decay.core], [-1, 1]).call(
                inputs
            )
            return ret  # tf.einsum("...ab,...cb->...ca", b, ret)
        else:
            raise NotImplementedError()


class EvalT2:
    """
    t^{u}

    """

    def __init__(self, decay, l):
        self.decay = decay
        self.l = l

    def __call__(self, inputs, cached=None):
        cached = {} if cached is None else cached
        p1 = inputs[f"{self.decay.outs[0]}_p"]
        p2 = inputs[f"{self.decay.outs[1]}_p"]
        p0 = p1 + p2  # inputs[f"{self.decay.core}_p"]
        # test = EvalP(self.decay.core, self.l)
        if self.l == 0:
            return tf.ones_like(p1[..., 0])
        elif self.l == 1:
            r = p1 - p2  # r^u
            # rt^u = - g^u_v r^v + (p1 - p2)^v (p1+p2)_v (p1+p2)^u
            r_t = (
                -r + ((mass2(p1) - mass2(p2)) / mass2(p0))[..., None] * p0
            )  #  * np.array([1,-1,-1,-1])
            # print( "diff", tf.reduce_sum(test(inputs) * r[...,None,:], axis=-1) - r_t)
            return r_t
        elif self.l == 2:
            r = p1 - p2
            g_t = (
                np.diag([1, -1, -1, -1])
                - p0[..., None] * p0[..., None, :] / mass2(p0)[..., None, None]
            )  # gbar^uv =  g^{uv} - p^u p^v / p^2
            # r_t_a =  -r + ((mass2(p1) - mass2(p2))/mass2(p0))[...,None] * p0
            r_t = tf.reduce_sum(
                g_t * r[..., None, :] * np.array([1, -1, -1, -1]), axis=-1
            )  # rt^u = gbar^uv g_vo r^o
            # t^uv =  rt^u rt^v + 1/3 r_t^2 gbar^uv
            ret = (
                r_t[..., None, :] * r_t[..., None]
                - 1 / 3 * dot(r_t, r_t)[..., None, None] * g_t
            )
            # print( "diff", tf.reduce_sum(test(inputs) * r[...,None,None,None,:] * r[...,None, None, :,None], axis=[-1,-2]) - ret)
            b = EvalBoost([self.decay.outs[0], self.decay.core], [-1, 1]).call(
                inputs
            )
            # print("boost matrix")
            # print(b)
            # print("origin t_2^{\\beta\\rho}", ret)
            b = tf.linalg.inv(b)  #  * np.array([1,-1,-1,-1])
            return tf.einsum("...ab,...bc->...ac", b, ret)
        else:
            raise NotImplementedError()


class EvalBoost:
    def __init__(self, boost, sign=None):
        self.boost = boost
        self.sign = sign

    def call(self, inputs, cached=None):
        from tf_pwa.angle import LorentzVector as lv

        mat = np.diag([1, 1, 1, 1])
        for i, j in zip(self.boost, self.sign):
            if j > 0:
                p = inputs[f"{i}_p"]
            else:
                p = lv.neg(inputs[f"{i}_p"])
            mat = tf.einsum("...ba,...bc->...ac", lv.boost_matrix(p), mat)
        return mat


class EvalP:
    """

    P^{u}_{v}

    """

    def __init__(self, core, l):
        self.core = core
        self.l = l

    def __call__(self, inputs, cached=None):
        cached = {} if cached is None else cached
        p0 = inputs[f"{self.core}_p"]
        if self.l == 0:
            return tf.ones_like(p0[..., 0])
        elif self.l == 1:
            # g^{uv}
            g_t = (
                np.diag([1, -1, -1, -1])
                - p0[..., None] * p0[..., None, :] / mass2(p0)[..., None, None]
            )
            # - g^u_v
            return -g_t * np.array([1, -1, -1, -1])
        elif self.l == 2:
            g_t = (
                np.diag([1, -1, -1, -1])
                - p0[..., None] * p0[..., None, :] / mass2(p0)[..., None, None]
            )  # gbar^uv =  g^{uv} - p^u p^v / p^2
            ret = (
                1
                / 2
                * (
                    g_t[..., None, :, None] * g_t[..., None, :, None, :]
                    + g_t[..., None, None, :] * g_t[..., None, :, :, None]
                )
                - 1 / 3 * g_t[..., None, None, :, :] * g_t[..., None, None]
            )
            ret = (
                ret
                * np.array([[1], [-1], [-1], [-1]])
                * np.array([1, -1, -1, -1])
            )
            # print(ret.shape)
            return ret
        else:
            raise NotImplementedError()
        return None


@register_decay_chain("cov_ten")
class CovTenDecayChain(DecayChain):
    def init_params(self, name=""):
        super().init_params(name)
        self.all_amp = self.build_einsum()

    def build_einsum(self):
        idx_map = IndexMap()
        all_ls = [i.get_ls_list() for i in self]
        all_amp = {}
        for i in itertools.product(*all_ls):
            all_amp[i] = self.build_decay_einsum(i)
        return all_amp

    def build_decay_einsum(self, ls, idx_map=None):
        all_rules = {}
        for (l, s), decay in zip(ls, self):
            rule1 = self.build_s_einsum(decay, l, s, idx_map)
            rule2 = self.build_l_einsum(decay, l, s, idx_map)
            rule2.set_inputs(f"{decay}_s_amp", rule1)
            rule2.set_inputs(
                f"{decay}_l_amp",
                EinSumCall(
                    f"{decay}_l_amp",
                    [f"{decay}_l_lorentz_{i}" for i in range(l)],
                    EvalT(decay, l),
                ),
            )
            all_rules[decay.core] = rule2
            # print(rule2)
        for i in self:
            for j in i.outs:
                if j in all_rules:
                    a = all_rules[j]
                    name = a.name
                    idx = a.index
                    a.name = name + "_pj"
                    a.replace_axis = {
                        k + "_pj": v for k, v in a.replace_axis.items()
                    }
                    for k in idx:
                        a.replace_axis[k] = k + "_pj"
                    a.index = [k + "_pj" for k in idx]
                    evalp = EinSumCall(
                        f"{j}_amp_pj_p",
                        [f"{j}_lorentz_{k}" for k in range(len(idx))]
                        + [f"{j}_lorentz_{k}_pj" for k in range(len(idx))],
                        EvalP(j, len(idx)),
                    )
                    tmp = EinSum(name, idx, [evalp, a])
                    all_rules[i.core].set_inputs(f"{j}_amp", tmp)
        ret = all_rules[self.top]
        phi = EinSum(
            f"{self.top}_amp_conj",
            [f"{self.top}_lorentz_{i}" for i in range(int(self.top.J))],
        )
        ret = EinSum("total", [], [phi, ret])
        ret.insert_extra_axis(f"{self.top}_amp_conj", [f"{self.top}_helicity"])
        for i in self.outs:
            ret.insert_extra_axis(f"{i}_amp", [f"{i}_helicity"])
        return ret

    def build_coupling_einsum(self, a, b, c, na, nb, nc, idx_map):
        idx_a = [f"{a}_lorentz_{i}" for i in range(na)]
        idx_b = [f"{b}_lorentz_{i}" for i in range(nb)]
        idx_c = [f"{c}_lorentz_{i}" for i in range(nc)]
        n = nb + nc - na
        if n % 2 == 0:
            k = n // 2
            gs = []
            for i in range(k):
                gs.append((idx_b[i], idx_c[i]))
            replace_axis = []
            for i in range(nb - k):
                replace_axis.append((idx_a[i], idx_b[k + i]))
            for i in range(nc - k):
                replace_axis.append((idx_a[nb - k + i], idx_c[i]))
            g = [EinSum("g", (i, j)) for i, j in gs]
            ret = EinSum(
                f"{a}_amp",
                idx_a,
                g + [EinSum(f"{b}_amp", idx_b), EinSum(f"{c}_amp", idx_c)],
                replace_axis=replace_axis,
            )
        else:
            idx_p = f"{a}_p"
            k = (n - 1) // 2
            gs = []
            for i in range(k):
                gs.append((idx_b[i], idx_c[i]))
            replace_axis = []
            for i in range(nb - 1 - k):
                replace_axis.append((idx_a[i], idx_b[k + i]))
            for i in range(nc - k - 1):
                replace_axis.append((idx_a[nb - 1 - k + i], idx_c[k + i]))
            epsilon = EinSum(
                "epsilon", (idx_a[-1], "delta", idx_b[-1], idx_c[-1])
            )
            if isinstance(a, str):
                aa = a.split("->")[0]
                p = EinSum(f"{aa}_p", ["delta"])
            else:
                p = EinSum(f"{a}_p", ["delta"])
            g = [EinSum("g", (i, j)) for i, j in gs]
            ret = EinSum(
                f"{a}_amp",
                idx_a,
                [epsilon, p]
                + g
                + [EinSum(f"{b}_amp", idx_b), EinSum(f"{c}_amp", idx_c)],
                replace_axis=replace_axis,
            )
        return ret

    def build_s_einsum(self, decay, l, s, idx_map):
        a = decay.core
        b, c = decay.outs
        ja, jb, jc = a.J, b.J, c.J
        na, nb, nc = int(s), int(jb), int(jc)
        return self.build_coupling_einsum(
            f"{decay}_s", b, c, s, nb, nc, idx_map
        )

    def build_l_einsum(self, decay, l, s, idx_map):
        a = decay.core
        b, c = decay.outs
        ja, jb, jc = a.J, b.J, c.J
        na, nb, nc = int(ja), int(jb), int(jc)
        return self.build_coupling_einsum(
            a, f"{decay}_l", f"{decay}_s", na, l, s, idx_map
        )

    def get_m_dep_list(self, data_c, data_p, all_data=None):
        def _prod(ls):
            ret = 1
            for i in ls:
                ret *= i
            return ret

        amp_m = self.get_m_dep(data_c, data_p, all_data=all_data)
        n_data = data_shape(data_p)
        tmp = amp_m[0]
        if tmp.shape[0] == 1:
            tmp = tf.tile(tmp, [n_data] + [1] * (len(tmp.shape) - 1))
        tmp = tf.reshape(tmp, (-1, _prod(tmp.shape[1:])))
        for j in amp_m[1:]:
            tmp2 = tf.reshape(j, (-1, _prod(j.shape[1:])))
            tmp = tmp[:, :, None] * tmp2[:, None, :]
            tmp = tf.reshape(tmp, (-1, _prod(tmp.shape[1:])))
        return tmp

    def get_amp(
        self, data_c, data_p, all_data=None, base_map=None, idx_map=None
    ):
        init_amp = self.build_wave_function(self.top, data_p[self.top]["p"])
        if init_amp.dtype in [tf.complex64, tf.complex128]:
            init_amp = tf.math.conj(init_amp)
        finals_amp = self.get_finals_amp(data_p)
        inputs = {f"{k}_amp": v for k, v in zip(self.outs, finals_amp)}
        inputs[f"{self.top}_amp_conj"] = init_amp
        for i in self.outs:
            inputs[f"{i}_p"] = data_p[i]["p"]
        inputs[f"{self.top}_p"] = data_p[self.top]["p"]
        for i in self.inner:
            inputs[f"{i}_p"] = data_p[i]["p"]
        amps = tf.stack(
            [i.call(inputs) for i in self.all_amp.values()], axis=-1
        )
        amp_m = self.get_m_dep_list(data_c, data_p, all_data=all_data)
        # amp_m = tf.stack(amp_m, axis=0)
        # print(amp_m.shape, amps.shape)
        for i in range(len(self.outs) + 1):
            amp_m = amp_m[..., None, :]
        amp_s = tf.reduce_sum(tf.cast(amps, amp_m.dtype) * amp_m, axis=-1)
        ret = amp_s  # tf.cast(amp_s, amp_p.dtype) * amp_p
        return ret

    def get_finals_amp(self, data_p):
        ret = []
        for i in self.outs:
            pi = data_p[i]["p"]
            ret.append(self.build_wave_function(i, pi))
        return ret

    def build_wave_function(self, particle, pi):
        if particle.J == 0:
            return tf.ones_like(pi[..., 0:1])
        if particle.J == 1:
            ret = wave_function(1, pi)
            if len(particle.spins) != 3:
                ret = tf.gather(
                    ret,
                    [_spin_int(i + particle.J) for i in particle.spins],
                    axis=-1,
                )
            return ret
        raise NotImplementedError()


def wave_function(J, p):
    if J == 0:
        return tf.ones_like(p[..., 0:1])
    if J == 1:
        beta = p[..., 1:] / p[..., 0:1]
        gamma = p[..., 0] / tf.sqrt(mass2(p))
        beta = tf.unstack(beta, axis=-1)
        dom = (1 + gamma) / gamma**2
        epsilon_0 = tf.stack(
            [
                gamma * beta[2],
                beta[2] * beta[0] / dom,
                beta[2] * beta[1] / dom,
                1 + beta[2] * beta[2] / dom,
            ],
            axis=-1,
        )
        zeros = tf.zeros_like(gamma)
        epsilon_0 = tf.complex(epsilon_0, tf.zeros_like(epsilon_0))
        epsilon_1 = tf.stack(
            [
                tf.complex(gamma * beta[0], gamma * beta[1]),
                tf.complex(
                    1 + beta[0] * beta[0] / dom, beta[0] * beta[1] / dom
                ),
                tf.complex(
                    beta[1] * beta[0] / dom, 1 + beta[1] * beta[1] / dom
                ),
                tf.complex(beta[2] * beta[0] / dom, beta[1] * beta[2] / dom),
            ],
            axis=-1,
        ) / math.sqrt(2)
        return tf.stack(
            [tf.math.conj(epsilon_1), epsilon_0, -epsilon_1], axis=-1
        )


class CovTenDecayNew(HelicityDecay):
    """
    Decay Class for covariant tensor formula
    """

    def __init__(self, *args, **kwargs):
        self.scheme = 1
        self.add_normal_factor = False
        super().__init__(*args, **kwargs)
        if "has_ql" not in kwargs:
            self.has_ql = False
        self.m0_zero = False
        if not hasattr(self, "m1_zero"):
            self.m1_zero = self.outs[0].mass == 0.0
        if not hasattr(self, "m2_zero"):
            self.m2_zero = self.outs[1].mass == 0.0
        self.decay_chain_params = {"aligned": False}

    def get_amp(self, data, data_p, **kwargs):
        ret = self.get_all_amp(data, data_p, **kwargs)
        m_dep = self.get_ls_amp(data, data_p, **kwargs)
        ret = tf.reduce_sum(ret * m_dep[..., None, None, None, :], axis=-1)
        for p, idx in zip([self.core, *self.outs], [-3, -2, -1]):
            if len(p.spins) > 0 and len(p.spins) != _spin_int(p.J * 2 + 1):
                indices = [_spin_int(i + p.J) for i in p.spins]
                ret = tf.gather(ret, axis=idx, indices=indices)
        return ret

    def get_angle_amp(self, data, data_p, **kwargs):
        total_ls_amp = self.get_all_amp(data, data_p, **kwargs)
        ret = tf.reduce_sum(total_ls_amp, axis=-1)
        for p, idx in zip([self.core, *self.outs], [-3, -2, -1]):
            if len(p.spins) > 0 and len(p.spins) != _spin_int(p.J * 2 + 1):
                indices = [_spin_int(i + p.J) for i in p.spins]
                ret = tf.gather(ret, axis=idx, indices=indices)
        return ret


@register_decay("cov_ten_simple")
class CovTenDecaySimple(CovTenDecayNew):
    """
    Decay Class for covariant tensor formula
    """

    def init_params(self):
        super().init_params()

    def get_amp(self, data, data_p, **kwargs):
        ret = self.get_all_amp(data, data_p, **kwargs)
        m_dep = self.get_ls_amp(data, data_p, **kwargs)
        ret = tf.reduce_sum(ret * m_dep[..., None, None, None, :], axis=-1)
        for p, idx in zip([self.core, *self.outs], [-3, -2, -1]):
            if len(p.spins) > 0 and len(p.spins) != _spin_int(p.J * 2 + 1):
                indices = [_spin_int(i + p.J) for i in p.spins]
                ret = tf.gather(ret, axis=idx, indices=indices)
        return ret

    def get_all_amp(self, data, data_p, **kwargs):
        # print(self)
        p1 = data_p[self.outs[0]]["p"]
        p2 = data_p[self.outs[1]]["p"]

        ret_list = []
        # ret = 0
        for i, (l, s) in enumerate(self.get_total_ls_list()):
            if self.ls_index is not None and i not in self.ls_index:
                continue
            from tf_pwa.cov_ten_ir import PWFA

            mu1 = 0 if self.m1_zero else 1
            mu2 = 0 if self.m2_zero else 1
            ampi = PWFA(
                p1,
                self.m1_zero,
                self.outs[0].J,
                p2,
                self.m2_zero,
                self.outs[1].J,
                self.core.J,
                s,
                l,
            )
            ret_list.append(ampi)
        ret = tf.stop_gradient(tf.stack(ret_list, axis=-1))
        return ret
