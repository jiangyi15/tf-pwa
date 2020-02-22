"""
Basic Amplitude Calculations.
A partial wave analysis process has following structure:

DecayGroup: addition
    DecayChain: multiplication
        Decay, Particle(Propagator)

"""

import functools
import numpy as np
import contextlib
from opt_einsum import contract_path, contract
from pprint import pprint
import copy
# from pysnooper import snoop

from .particle import Decay, Particle as BaseParticle, DecayChain as BaseDecayChain, DecayGroup as BaseDecayGroup
from .tensorflow_wrapper import tf
from .cal_angle import prepare_data_from_decay, split_generator
from .breit_wigner import barrier_factor2 as barrier_factor, BWR, BW
from .dfun import get_D_matrix_lambda
from .cg import cg_coef
from .variable import VarsManager, Variable
from .data import data_shape, split_generator, data_to_tensor, data_map

from .config import regist_config, get_config, temp_config
from .einsum import einsum


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
    return Variable(name, shape, is_complex, **kwargs)


@contextlib.contextmanager
def variable_scope(vm=None):
    if vm is None:
        vm = VarsManager(dtype=get_config("dtype"))
    with temp_config("vm", vm):
        yield vm


def simple_deepcopy(dic):
    if isinstance(dic, dict):
        return {k: simple_deepcopy(v) for k, v in dic.items()}
    if isinstance(dic, list):
        return [simple_deepcopy(v) for v in dic]
    if isinstance(dic, tuple):
        return tuple([simple_deepcopy(v) for v in dic])
    return dic


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

    def init_params(self):
        self.d = 3.0
        if self.mass is None:
            self.mass = add_var(self, "mass", fix=True)
        #else:
        #    self.mass = add_var(self, "mass", value=self.mass, fix=True)
        if self.width is None:
            self.width = add_var(self, "width", fix=True)
        #else:
        #    self.width = add_var(self, "width", value=self.width, fix=True)

    def get_amp(self, data, data_c=None):
        mass = self.get_mass()
        width = self.get_width()
        if data_c is None:
            ret = BW(data["m"], mass, width)
            return ret
        decay = self.decay[0]
        # return BW(data["m"], self.mass, self.width)
        q = data_c["|q|"]
        q0 = data_c["|q0|"]
        ret = BWR(data["m"], mass, width, q, q0, min(decay.get_l_list()), self.d)
        return ret  # tf.convert_to_tensor(complex(1.0), dtype=get_config("complex_dtype"))

    def amp_shape(self):
        return ()

    def get_mass(self):
        if callable(self.mass):
            return self.mass()
        return self.mass

    def get_width(self):
        if callable(self.width):
            return self.width()
        return self.width


class HelicityDecay(Decay):

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
            return p.get_mass()

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

        g_ls = tf.stack(self.g_ls())
        q0 = self.get_relative_momentum(data_p, False)
        data["|q0|"] = q0
        if "|q|" in data:
            q = data["|q|"]
        else:
            q = self.get_relative_momentum(data_p, True)
            data["|q|"] = q
        bf = barrier_factor(self.get_l_list(), q, q0, self.d)
        mag = g_ls
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
        H = self.get_helicity_amp(data, data_p)
        H = tf.reshape(H, (-1, 1, len(self.outs[0].spins), len(self.outs[1].spins)))
        H = tf.cast(H, dtype=D_conj.dtype)
        ret = H * D_conj
        # print(self, H, D_conj)
        # exit()
        aligned = False
        if aligned:
            for j, particle in enumerate(self.outs):
                if particle.J != 0 and "aligned_angle" in data[particle]:
                    ang = data[particle].get("aligned_angle", None)
                    if ang is None:
                        continue
                    dt = get_D_matrix_lambda(ang, particle.J, particle.spins, particle.spins)
                    dt_shape = [-1, 1, 1, 1, 1]
                    dt_shape[j + 2] = len(particle.spins)
                    dt_shape[j + 3] = len(particle.spins)
                    dt = tf.reshape(dt, dt_shape)
                    D_shape = [-1, len(a.spins), len(b.spins), len(c.spins)]
                    D_shape.insert(j + 3, 2)
                    D_shape[j + 3] = 1
                    ret = tf.reshape(ret, D_shape)
                    ret = dt * ret
                    ret = tf.reduce_sum(ret, axis=j + 2)
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


class HelicityDecayNP(HelicityDecay):
    def init_params(self):
        a = self.outs[0].spins
        b = self.outs[1].spins
        self.H = add_var(self, "H", is_complex=True, shape=(len(a), len(b)))

    def get_helicity_amp(self, data, data_p):
        return tf.stack(self.H())


def get_parity_term(j1, p1, j2, p2, j3, p3):
    p = p1 * p2 * p3 * (-1) ** (j1 - j2 - j3)
    return p


class HelicityDecayP(HelicityDecay):
    def init_params(self):
        a = self.core
        b = self.outs[0]
        c = self.outs[1]
        n_b = len(b.spins)
        n_c = len(c.spins)
        self.parity_term = get_parity_term(a.J, a.P, b.J, b.P, c.J, c.P)
        if n_b != 1:
            self.H = add_var(self, "H", is_complex=True, shape=((n_b+1) // 2, n_c))
            self.part_H = 0
        else:
            self.H = add_var(self, "H", is_complex=True, shape=(n_b, (n_c+1) // 2))
            self.part_H = 1

    def get_helicity_amp(self, data, data_p):
        n_b = len(self.outs[0].spins)
        n_c = len(self.outs[1].spins)
        H_part = tf.stack(self.H())
        if self.part_H == 0:
            H = tf.concat([H_part, self.parity_term * H_part[(n_b - 2) // 2::-1]], axis=0)
        else:
            H = tf.concat([H_part, self.parity_term * H_part[:, (n_c - 2) // 2::-1]], axis=1)
        return H


class DecayChain(BaseDecayChain):
    """A list of Decay as a chain decay"""

    def __init__(self, *args, **kwargs):
        super(DecayChain, self).__init__(*args, **kwargs)

    def init_params(self):
        self.total = add_var(self, "total", is_complex=True)
        self.aligned = True

    def get_amp_total(self):
        return self.total()

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
            if len(i.decay) == 1 and i.decay[0] in self:
                data_c_i = data_c[i.decay[0]]
                if "|q|" not in data_c_i:
                    data_c_i["|q|"] = i.decay[0].get_relative_momentum(data_p, True)
                if "|q0|" not in data_c_i:
                    data_c_i["|q0|"] = i.decay[0].get_relative_momentum(data_p, False)
                amp_p.append(i.get_amp(data_p[i], data_c_i))
            else:
                amp_p.append(i.get_amp(data_p[i]))
        rs = self.get_amp_total() * tf.reduce_sum(amp_p, axis=0)
        amp_d[0] = rs

        if self.aligned:
            for i in self:
                for j in i.outs:
                    if j.J != 0 and "aligned_angle" in data_c[i][j]:
                        ang = data_c[i][j]["aligned_angle"]
                        dt = get_D_matrix_lambda(ang, j.J, j.spins, j.spins)
                        amp_d.append(dt)
                        idx = [base_map[j], base_map[j].upper()]
                        indices.append(idx)
                        final_indices = final_indices.replace(*idx)
        idxs = []
        for i in indices:
            tmp = "".join(iter_idx + i)
            idxs.append(tmp)
        idx = ",".join(idxs)
        idx_s = "{}->{}".format(idx, final_indices)
        # ret = amp * tf.reshape(rs, [-1] + [1] * len(self.amp_shape()))
        ret = einsum(idx_s, *amp_d)
        # print(self, ret[0])
        # exit()
        # ret = einsum(idx_s, *amp_d)
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
    """ A Group of Decay Chains with the same final particles."""

    def __init__(self, chains):
        self.chains_idx = list(range(len(chains)))
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

        used_chains = tuple([self.chains[i] for i in self.chains_idx])
        chain_maps = self.get_chains_map(used_chains)
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

    def sum_amp(self, data, cached=True):
        if not cached:
            data = simple_deepcopy(data)
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

    def get_res_map(self):
        res_map = {}
        for i, decay in enumerate(self.chains):
            for j in decay.inner:
                if j not in res_map:
                    res_map[j] = []
                res_map[j].append(i)
        return res_map

    def set_used_res(self, res, only=False):
        res_set = set()
        for i in res:
            if isinstance(i, str):
                res_set.add(BaseParticle(i))
            elif isinstance(i, BaseParticle):
                res_set.add(i)
            else:
                raise TypeError("type({}) = {} not a Particle".format(i, type(i)))
        if not only:
            used_res = set()
            for i in res_set:
                for j, c in enumerate(self.chains):
                    if i in c.inner:
                        used_res.add(j)
            self.set_used_chains(list(used_res))
        else:
            unused_res = set(self.resonances) - res_set
            unused_decay = set()
            res_map = self.get_res_map()
            for i in unused_res:
                for j in res_map[i]:
                    unused_decay.add(j)
            used_decay = []
            for i, _ in enumerate(self.chains):
                if i not in unused_decay:
                    used_decay.append(i)
            self.set_used_chains(used_decay)

    def set_used_chains(self, used_chains):
        self.chains_idx = list(used_chains)

    def partial_weight(self, data, combine=None):
        chains = list(self.chains)
        if combine is None:
            combine = [[i] for i in range(len(chains))]
        o_used_chains = self.chains_idx
        weights = []
        for i in combine:
            self.set_used_chains(i)
            weight = self.sum_amp(data)
            weights.append(weight)
        self.set_used_chains(o_used_chains)
        return weights

    def generate_phasespace(self, num=100000):

        def get_mass(i):
            mass = i.get_mass()
            if mass is None:
                raise Exception("mass is required for particle {}".format(i))
            return mass

        top_mass = get_mass(self.top)
        final_mass = [get_mass(i) for i in self.outs]
        from .phasespace_tf import PhaseSpaceGenerator
        a = PhaseSpaceGenerator(top_mass, final_mass)
        data = a.generate(num)
        return dict(zip(self.outs, data))


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
        res = decay_group.resonances
        self.used_res = res
        self.res = res

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

    def set_used_res(self, res):
        self.decay_group.set_used_res(res)

    def set_used_chains(self, used_chains):
        self.decay_group.set_used_chains(used_chains)

    def partial_weight(self, data, combine=None):
        return self.decay_group.partial_weight(data, combine)

    def get_params(self,trainable_only=False):
        return self.vm.get_all_dic(trainable_only)

    def set_params(self, var):
        self.vm.set_all(var)

    @property
    def variables(self):
        return self.vm.variables

    @property
    def trainable_variables(self):
        return self.vm.trainable_variables

    def __call__(self, data, cached=False):
        return self.decay_group.sum_amp(data)
