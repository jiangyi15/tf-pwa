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
import contextlib
from opt_einsum import contract
from pprint import pprint
import copy
from pysnooper import snoop

from .particle import Decay, Particle as BaseParticle, DecayChain as BaseDecayChain, DecayGroup as BaseDecayGroup
from .tensorflow_wrapper import tf
from .data import prepare_data_from_decay, split_generator
from .breit_wigner import barrier_factor2 as barrier_factor, BWR, BW
from .dfun import get_D_matrix_lambda
from .cg import cg_coef
from .variable import VarsManager, Variable
from .data import data_shape, split_generator, data_to_tensor, data_map

from .config import regist_config, get_config, temp_config


def data_device(data):
    
    def get_device(dat):
        if hasattr(dat, "device"):
            return dat.device
        return None
    
    pprint(data_map(data, get_device))
    return data


def get_name(self, names):
    name = (str(self) + "_" + names) \
        .replace(":", "/").replace("+", ".") \
        .replace(",", "").replace("[", "").replace("]", "").replace(" ", "")
    return name


def add_var(self, names, is_complex=False, shape=(), **kwargs):
    name = get_name(self, names)
    return Variable(name, shape, is_complex,**kwargs)


def einsum(eins, *args, **kwargs):
    has_ellipsis = False
    if "..." in eins:
        has_ellipsis = True
        eins = eins.replace("...","I")
    inputs, final = eins.split("->")
    idx = inputs.split(",")
    order = "".join(idx)
    shapes = [list(i.shape) for i in args]
    idx_size = {}
    for i, j in zip(idx,shapes):
        for k, v in zip(i,j):
            idx_size[k] = v
    return tf.einsum(eins, *args, **kwargs)


@contextlib.contextmanager
def variable_scope(vm=None):
    if vm is None:
        vm = VarsManager(dtype=get_config("dtype"))
    with temp_config("vm", vm):
        yield vm


class Var(object):
    def __init__(self, name):
        self.name = name

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __get__(self, instance, var):
        value = instance.__dict__[self.name]
        if callable(value):
            return value()
        return value


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
    mass = Var("mass")
    width = Var("width")

    def __init__(self, *args, **kwargs):
        super(Particle, self).__init__(*args, **kwargs)

    def init_params(self):
        if self.mass is None:
            self.mass = add_var(self, "mass", trainable=True)
        if self.width is None:
            self.width = add_var(self, "width")

    def get_amp(self, data, data_c=None):
        if data_c is None:
            return tf.convert_to_tensor(complex(1.0), dtype=get_config("complex_dtype"))
        decay = self.decay[0]
        return BW(data["m"], self.mass, self.width)
        q = data_c["|q|"]
        q0 = data_c["|q0|"]
        ret = BWR(data["m"], self.mass, self.width, q, q0, min(decay.get_l_list()), decay.d)
        return ret  # tf.convert_to_tensor(complex(1.0), dtype=get_config("complex_dtype"))

    def amp_shape(self):
        return ()


class HelicityDecay(Decay):
    g_ls = Var("g_ls")

    def __init__(self, *args, **kwargs):
        super(HelicityDecay, self).__init__(*args, **kwargs)

    def init_params(self):
        self.d = 3.0
        ls = self.get_ls_list()
        self.g_ls = add_var(self, "g_ls", is_complex=True, shape=(len(ls),))

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
        ret = np.zeros(shape=(m, *n))
        for i, ls_i in enumerate(ls):
            l, s = ls_i
            for i1, lambda_b in enumerate(range(-jb, jb + 1)):
                for i2, lambda_c in enumerate(range(-jc, jc + 1)):
                    ret[i][i1][i2] = np.sqrt((2 * l + 1) / (2 * ja + 1)) \
                                * cg_coef(jb, jc, lambda_b, -lambda_c, s, lambda_b - lambda_c) \
                                * cg_coef(l, s, 0, lambda_b - lambda_c, ja, lambda_b - lambda_c)
        return tf.convert_to_tensor(ret)

    def get_helicity_amp(self, data, data_p):
        g_ls = tf.complex(*list(zip(* self.g_ls)))
        norm_r, norm_i = tf.math.real(g_ls), tf.math.imag(g_ls)
        q0 = self.get_relative_momentum(data_p, False)
        data["|q0|"] = q0
        if "|q|" in data:
            q = data["|q|"]
        else:
            q = self.get_relative_momentum(data_p, True)
            data["|q|"] = q
        bf = barrier_factor(self.get_l_list(), q, q0, self.d)
        mag = tf.complex(tf.cast(norm_r, bf.dtype), tf.cast(norm_i, bf.dtype))
        # meg = tf.reshape(meg, (-1, 1))
        m_dep = mag * tf.cast(bf, mag.dtype)
        cg_trans = tf.cast(self.get_cg_matrix(), m_dep.dtype)
        n_ls = len(self.get_ls_list())
        m_dep = tf.reshape(m_dep, (-1, n_ls, 1, 1))
        cg_trans = tf.reshape(cg_trans, (n_ls, len(self.outs[0].spins), len(self.outs[1].spins)))
        H = tf.reduce_sum(m_dep * cg_trans, axis=1)
        ret = tf.reshape(H, (-1, 1, len(self.outs[0].spins), len(self.outs[1].spins)))
        return ret

    def get_amp(self, data, data_p):
        a = self.core
        b = self.outs[0]
        c = self.outs[1]
        ang = data[b]["ang"]
        D_conj = get_D_matrix_lambda(ang, a.J, a.spins, b.spins, c.spins)
        amp_d = []
        for j in range(2):
            particle = self.outs[j]
            if particle.J != 0:
                ang = data[particle].get("aligned_angle", None)
                if ang is None:
                    continue
                dt = get_D_matrix_lambda(ang, particle.J, particle.spins, particle.spins)
                dt_shape = [-1, 1, 1, 1, 1]
                dt_shape[j+2] = len(particle.spins)
                dt_shape[j+3] = len(particle.spins)
                dt = tf.reshape(dt, dt_shape)
                D_shape = [-1, len(a.spins), len(b.spins), len(c.spins)]
                D_shape.insert(j+3, 1)
                D_conj = tf.reshape(D_conj, D_shape)
                D_conj = dt * D_conj
                D_conj = tf.reduce_sum(D_conj, axis=j+2)
        H = self.get_helicity_amp(data, data_p)
        H = tf.cast(H, dtype=D_conj.dtype)
        ret = H * D_conj
        return ret

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
    total = Var("total")

    def __init__(self, *args, **kwargs):
        super(DecayChain, self).__init__(*args, **kwargs)

    def init_params(self):
        self.total = add_var(self, "total", is_complex=True)

    def get_amp(self, data_c, data_p, base_map=None):
        base_map = self.get_base_map(base_map)
        iter_idx = ["..."]
        amp_d = [None]
        indices = [[]]
        final_indices = "".join(iter_idx + self.amp_index(base_map))
        for i in self:
            indices.append(i.amp_index(base_map))
            amp_d.append(i.get_amp(data_c[i], data_p))

        amp_p = []
        for i in self.inner:
            if len(i.decay) == 1:
                amp_p.append(i.get_amp(data_p[i], data_c[i.decay[0]]))
            else:
                amp_p.append(i.get_amp(data_p[i]))
        rs = tf.complex(*self.total) * tf.reduce_sum(amp_p, axis=0)
        amp_d[0] = rs
        
        idxs = []
        for i in indices:
            tmp = "".join(iter_idx + i)
            idxs.append(tmp)
        idx = ",".join(idxs)
        idx_s = "{}->{}".format(idx, final_indices)
        #ret = amp * tf.reshape(rs, [-1] + [1] * len(self.amp_shape()))
        # ret = contract(idx_s, *amp_d, backend="tensorflow")
        ret = einsum(idx_s, *amp_d)
        return ret

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
        self.init_params()

    def init_params(self):
        for i in self.resonances:
            i.init_params()
        inited_set = set()
        for i in self:
            i.init_params()
            for j in i:
                if j not in inited_set:
                    j.init_params()
                    inited_set.add(j)

    def get_amp(self, data):
        data_particle = data["particle"]
        data_decay = data["decay"]

        chain_maps = self.get_chains_map()
        base_map = self.get_base_map()
        ret = []
        amp_idx = self.amp_index(base_map)
        idx_ein = "".join(["..."] + amp_idx)
        for chains, data_d in zip(chain_maps, data_decay):
            for decay_chain in chains:
                data_c = rename_data_dict(data_d, chains[decay_chain])
                data_p = rename_data_dict(data_particle, chains[decay_chain])
                amp = decay_chain.get_amp(data_c, data_p, base_map=base_map)
                ret.append(amp)
        ret = tf.reduce_sum(ret, axis=0)
        return ret

    # @tf.function(experimental_relax_shapes=True)
    def sum_amp(self, data):
        data = data_to_tensor(data)
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


def value_and_grad(f, var):
    with tf.GradientTape() as tape:
        s = f(var)
    g = tape.gradient(s, var)
    return s, g


class AmplitudeModel(object):
    def __init__(self, decay_group):
        self.decay_group = decay_group
        with variable_scope() as vm:
            decay_group.init_params()
        self.vm = vm

    def cache_data(self, data, split=None, batch=None):
        for i in self.decay_group:
            for j in i.inner:
                print(j)
        if split is None and batch is None:
            return data
        else:
            n = data_shape(data)
            if batch is None:  # split个一组，共batch组
                batch = (n + split - 1) // split
            ret = list(split_generator(data, batch))
            return ret

    def get_params(self):
        return self.vm.get_all()

    def set_params(self, var):
        self.vm.set_all(var)

    @property
    def variables(self):
        return self.vm.get_all()

    @property
    def trainable_variables(self):
        return self.vm.get_all()

    def __call__(self, data, cached=False):
        return self.decay_group.sum_amp(data)


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
    print(get_config("vm").get_all())
    for i in data_s:
        def f(var):
            return tf.reduce_sum(de.sum_amp(i))
        s, g = value_and_grad(f, get_config("vm").get_all())
        gs.append(g)
        ss.append(s)
    print(time.time() - a)
    print(list(map(sum, zip(*gs))))
    print(sum(ss))
    return tf.reduce_sum(ret)
