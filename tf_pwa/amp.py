"""
Basic Amplitude Calculations.
A partial wave analysis process has following structure:

DecayGroup: addition
    DecayChain: multiplication
        Decay, Particle(Propagator)

TODO: using indices order to reduce transpose
"""

import functools
import numpy as np

from .particle import Decay, Particle as BaseParticle, DecayChain as BaseDecayChain, DecayGroup as BaseDecayGroup
from .tensorflow_wrapper import tf
from .data import prepare_data_from_decay, split_generator
from .breit_wigner import barrier_factor, BW
from .dfun import get_D_matrix_lambda
from .cg import cg_coef

all_var = []


def add_var(self, var, *args, trainable=False, **kwargs):
    ret = tf.Variable(var, *args, dtype="float64", trainable=trainable, **kwargs)
    if trainable:
        all_var.append(ret)
    return ret


def simple_cache_fun(f):
    name = "simple_cached_" + f.__name__

    @functools.wraps(f)
    def g(self, *args, **kwargs):
        if not hasattr(self, name):
            setattr(self, name, f(self, *args, **kwargs))
        return getattr(self, name)

    return g


def get_relative_p(m_0, m_1, m_2):
    M12S = m_1 + m_2
    M12D = m_1 - m_2
    p = (m_0 - M12S) * (m_0 + M12S) * (m_0 - M12D) * (m_0 + M12D)
    q = (p + tf.abs(p)) / 2  # if p is negative, which results from bad data, the return value is 0.0
    return tf.sqrt(q) / (2 * m_0)


class Particle(BaseParticle):
    def __init__(self, *args, **kwargs):
        super(Particle, self).__init__(*args, **kwargs)
        self.init_params()

    def init_params(self):
        if self.mass is None:
            self.mass = add_var(self, 1.0, trainable=True)
        if self.width is None:
            self.width = add_var(self, 1.0)

    def get_amp(self, data, data_c=None):
        m = data["m"]
        m0 = self.mass
        g0 = self.width
        return BW(m, m0, g0)

    def amp_shape(self):
        return ()


class HelicityDecay(Decay):
    def __init__(self, *args, **kwargs):
        super(HelicityDecay, self).__init__(*args, **kwargs)
        self.init_params()

    def init_params(self):
        self.d = 3.0
        ls = self.get_ls_list()
        self.g_ls_r = add_var(self, tf.ones(shape=(len(ls)), dtype="float64"), trainable=True)
        self.g_ls_i = add_var(self, tf.ones(shape=(len(ls)), dtype="float64"), trainable=True)

    def get_relative_momentum(self, data, from_data=False):

        def _get_mass(p):
            if from_data or p.mass is None:
                return data[p]["m"]
            return p.mass

        m0 = _get_mass(self.core)
        m1 = _get_mass(self.outs[0])
        m2 = _get_mass(self.outs[1])
        return get_relative_p(m0, m1, m2)

    @functools.lru_cache()
    def get_cg_matrix(self):  # CG factor inside H
        """
        [(l,s),(lambda_b,lambda_c)]

        .. math::
          \\sqrt{\\frac{ 2 l + 1 }{ 2 j_a + 1 }}
          \\langle j_b, j_c, \\lambda_b, - \\lambda_c | s, \\lambda_b - \\lambda_c \\rangle
          \\langle l, s, 0, \\lambda_b - \\lambda_c | j_a, \\lambda_b - \\lambda_c \\rangle
        """
        ls = self.get_ls_list()
        m = len(ls)
        ja = self.core.J
        jb = self.outs[0].J
        jc = self.outs[1].J
        n = (2 * jb + 1), (2 * jc + 1)
        ret = np.zeros(shape=(*n, m))
        for i, ls_i in enumerate(ls):
            l, s = ls_i
            for i1, lambda_b in enumerate(range(-jb, jb + 1)):
                for i2, lambda_c in enumerate(range(-jc, jc + 1)):
                    ret[i1][i2][i] = np.sqrt((2 * l + 1) / (2 * ja + 1)) \
                                * cg_coef(jb, jc, lambda_b, -lambda_c, s, lambda_b - lambda_c) \
                                * cg_coef(l, s, 0, lambda_b - lambda_c, ja, lambda_b - lambda_c)
        return ret

    def get_helicity_amp(self, data, data_p, params):
        norm_r, norm_i = self.g_ls_r, self.g_ls_i
        q0 = self.get_relative_momentum(data_p, False)
        data["|q0|"] = q0
        if "|q|" in data:
            q = data["|q|"]
        else:
            q = self.get_relative_momentum(data_p, True)
            data["|q|"] = q
        bf = barrier_factor(self.get_l_list(), q, q0, self.d)
        meg = tf.complex(tf.cast(norm_r, bf.dtype), tf.cast(norm_i, bf.dtype))
        meg = tf.reshape(meg, (-1, 1))
        m_dep = meg * tf.cast(bf, meg.dtype)
        cg_trans = tf.cast(self.get_cg_matrix(), m_dep.dtype)
        H = tf.einsum("ijk,kl->lij", cg_trans, m_dep)
        ret = tf.reshape(H, (-1, 1, len(self.outs[0].spins), len(self.outs[1].spins)))
        return ret

    def get_amp(self, data, data_p, params=None):
        a = self.core
        b = self.outs[0]
        c = self.outs[1]
        ang = data[b]["ang"]
        ret = get_D_matrix_lambda(ang, a.J, a.spins, b.spins, c.spins)
        H = self.get_helicity_amp(data, data_p, params)
        H = tf.cast(H, dtype=ret.dtype)
        return H * ret

    def amp_shape(self):
        ret = [len(self.core.spins)]
        for i in self.outs:
            ret.append(len(i.spins))
        return tuple(ret)

    @simple_cache_fun
    def amp_index(self, base_map):
        ret = [base_map[self.core]]
        for i in self.outs:
            ret.append(base_map[i])
        return ret


class DecayChain(BaseDecayChain):
    def __init__(self, *args, **kwargs):
        super(DecayChain, self).__init__(*args, **kwargs)
        self.init_params()

    def init_params(self):
        self.total_r = tf.Variable(1.0, dtype="float64")
        self.total_i = tf.Variable(0.0, dtype="float64")

    def get_amp(self, data_c, data_p, params=None, base_map=None):
        base_map = self.get_base_map(base_map)
        amp_d = []
        indices = []
        for i in self:
            indices.append(i.amp_index(base_map))
            amp_d.append(i.get_amp(data_c[i], data_p, params))
        idxs = []
        for i in indices:
            tmp = "".join(["i"] + i)
            idxs.append(tmp)
        idx = ",".join(idxs)
        idx_s = "{}->{}".format(idx, "".join(["i"] + self.amp_index(base_map)))
        amp = tf.einsum(idx_s, *amp_d)
        amp_p = []
        for i in self.inner:
            if len(i.decay) <= 0:
                amp_p.append(i.get_amp(data_p[i]))
            else:
                amp_p.append(i.get_amp(data_p[i], data_c[i.decay[0]]))
        rs = tf.reduce_sum(amp_p, axis=0)
        ret = amp * tf.reshape(rs, [-1] + [1] * len(self.amp_shape()))
        return tf.complex(self.total_r, self.total_i) * ret

    def amp_shape(self):
        ret = [len(self.top.spins)]
        for i in self.outs:
            ret.append(len(i.spins))
        return tuple(ret)

    @simple_cache_fun
    def amp_index(self, base_map=None):
        if base_map is None:
            base_map = self.get_base_map()
        ret = [base_map[self.top]]
        for i in self.outs:
            ret.append(base_map[i])
        return ret

    def get_base_map(self, base_map=None):
        gen = index_generator(base_map)
        if base_map is None:
            base_map = {}
        ret = base_map.copy()
        if self.top not in base_map:
            ret[self.top] = next(gen)
        for i in self.outs:
            if i not in base_map:
                ret[i] = next(gen)
        for i in self.inner:
            if i not in ret:
                ret[i] = next(gen)
        return ret


class DecayGroup(BaseDecayGroup):
    def __init__(self, chains):
        first_chain = chains[0]
        if not isinstance(first_chain, DecayChain):
            chains = [DecayChain(i) for i in chains]
        super(DecayGroup, self).__init__(chains)

    def get_amp(self, data):
        data_particle = data["particle"]
        data_decay = data["decay"]

        chain_maps = self.get_chains_map()
        base_map = self.get_base_map()
        ret = []
        amp_idx = self.amp_index(base_map)
        idx_ein = "".join(["i"] + amp_idx)
        for chains, data_d in zip(chain_maps, data_decay):
            amp_same = []
            for decay_chain in chains:
                data_c = rename_data_dict(data_d, chains[decay_chain])
                data_p = rename_data_dict(data_particle, chains[decay_chain])
                amp = decay_chain.get_amp(data_c, data_p, params={"m": np.array(2.0), "g": np.array(1.0)},
                                          base_map=base_map)
                amp_same.append(amp)
            amp_same = tf.reduce_sum(amp_same, axis=0)
            aligned = {}
            for dec in data_d:
                for j in dec.outs:
                    if j.J != 0 and "aligned_angle" in data_d[dec][j]:
                        ang = data_d[dec][j]["aligned_angle"]
                        dt = get_D_matrix_lambda(ang, j.J, j.spins, j.spins)
                        aligned[j] = dt
                        idx = base_map[j]
                        ein = "{},{}->{}".format(idx_ein, "i" + idx + idx.upper(),
                                                 idx_ein.replace(idx, idx.upper()))
                        amp_same = tf.einsum(ein, amp_same, dt)
            ret.append(amp_same)
        ret = tf.reduce_sum(ret, axis=0)
        return ret

    def sum_amp(self, data):
        amp = self.get_amp(data)
        amp2s = tf.math.real(amp * tf.math.conj(amp))
        idx = list(range(1, len(amp2s.shape)))
        sum_A = tf.reduce_sum(amp2s, idx)
        return sum_A

    @simple_cache_fun
    def amp_index(self, gen=None, base_map=None):
        if base_map is None:
            base_map = self.get_base_map()
        ret = [base_map[self.top]]
        for i in self.outs:
            ret.append(base_map[i])
        return ret

    def get_base_map(self, gen=None, base_map=None):
        if gen is None:
            gen = index_generator(base_map)
        if base_map is None:
            base_map = {self.top: next(gen)}
        for i in self.outs:
            base_map[i] = next(gen)
        return base_map


def index_generator(base_map=None):
    indices = "abcdefghjklmnopqrstuvwxyz"
    if base_map is not None:
        for i in base_map:
            indices = indices.replace(base_map[i], '')
    for i in indices:
        yield i


def rename_data_dict(data, idx_map):
    if isinstance(data, dict):
        return {idx_map.get(k, k): rename_data_dict(v, idx_map) for k, v in data.items()}
    if isinstance(data, tuple):
        return tuple([rename_data_dict(i, idx_map) for i in data])
    if isinstance(data, list):
        return [rename_data_dict(i, idx_map) for i in data]
    return data


def test_amp(fnames="data/data_test.dat"):
    a = Particle("A", J=1, P=-1, spins=(-1, 1))
    b = Particle("B", J=1, P=-1)
    c = Particle("C", J=0, P=-1)
    d = Particle("D", J=1, P=-1)
    bd = Particle("BD", 1, 1)
    cd = Particle("CD", 1, 1)
    bc = Particle("BC", 1, 1)
    R = Particle("R", 1, 1)
    dec1 = HelicityDecay(a, [bc, d])
    dec2 = HelicityDecay(bc, [b, c])
    dec3 = HelicityDecay(a, [cd, b])
    HelicityDecay(cd, [c, d])
    HelicityDecay(a, [bd, c])
    HelicityDecay(bd, [b, d])
    HelicityDecay(a, [R, c])
    HelicityDecay(R, [b, d])
    de = DecayGroup(a.chain_decay())
    data = prepare_data_from_decay(fnames, de)
    import time
    a = time.time()
    ret = de.sum_amp(data)
    print(time.time() - a)
    data_s = list(split_generator(data, 60000))
    a = time.time()
    gs = []
    ss = []
    for i in data_s:
        def f(var):
            return tf.reduce_sum(de.sum_amp(i))
        s, g = value_and_grad(f, all_var)
        gs.append(g)
        ss.append(s)
    print(time.time() - a)
    print(list(map(sum, zip(*gs))))
    print(sum(ss))
    return tf.reduce_sum(ret)


def value_and_grad(f, var):
    with tf.GradientTape() as tape:
        s = f(var)
    g = tape.gradient(s, var)
    return s, g