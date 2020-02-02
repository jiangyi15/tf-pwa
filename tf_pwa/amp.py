import functools
import numpy as np

from .particle import Decay, Particle as BaseParticle, DecayChain as BaseDecayChain, DecayGroup as BaseDecayGroup
from .tensorflow_wrapper import tf
from .data import prepare_data_from_decay
from .breit_wigner import barrier_factor as default_barrier_factor, BW
from .dfun import get_D_matrix_lambda

def simple_cache_fun(f):
  name = "simple_cached_"+f.__name__
  @functools.wraps(f)
  def g(self, *args, **kwargs):
    if not hasattr(self,name):
      setattr(self, name, f(self, *args, **kwargs))
    return getattr(self, name)
  return g


class Particle(BaseParticle):
    def get_amp(self, data, params=None):
        if self.mass:
            m0 = self.mass
        else:
            m0 = params["m"]
        if self.width is not None:
            g0 = self.width
        else:
            g0 = params["g"]
        m = data["m"]
        return BW(m, m0, g0)
    
    def amp_shape(self):
        return ()


class HelicityDecay(Decay):
    def get_amp(self, data, data_p, params=None):
        a = self.core
        b = self.outs[0]
        c = self.outs[1]
        ang = data[b]["ang"]
        ret = get_D_matrix_lambda(ang, a.J, a.spins, b.spins, c.spins)
        H = tf.ones((len(a.spins),len(b.spins),len(c.spins)),dtype=ret.dtype)
        return H * ret
    def amp_shape(self):
        ret = [len(self.core.spins)]
        for i in self.outs:
            ret.append(len(i.spins))
        return tuple(amp_shape)
    @simple_cache_fun
    def amp_index(self, base_map):
        ret = [base_map[self.core]]
        for i in self.outs:
            ret.append(base_map[i])
        return ret

class DecayChain(BaseDecayChain):
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
            amp_p.append(i.get_amp(data_p[i], params))
        rs = tf.reduce_sum(amp_p, axis=0)
        return amp
    def amp_shape(self):
        ret = [len(self.top.spins)]
        for i in self.outs:
            ret.append(len(i.spins))
        return tuple(ret)
    @simple_cache_fun
    def amp_index(self, base_map=None):
        if base_map is None:
            base_map = self.get_base_map()
        ret = []
        ret.append(base_map[self.top])
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
        amp_idx =  self.amp_index(base_map)
        idx_ein = "".join(["i"] + amp_idx)
        for chains, data_d in zip(chain_maps, data_decay):
            amp_same = []
            for decay_chain in chains:
                data_c = rename_data_dict(data_d, chains[decay_chain])
                data_p = rename_data_dict(data_particle, chains[decay_chain])
                amp = decay_chain.get_amp(data_c, data_p, params = {"m":np.array(2.0),"g":np.array(1.0)}, base_map=base_map)
                amp_same.append(amp)
            amp_same = tf.reduce_sum(amp_same, axis=0)
            aligned = {}
            for dec in data_d:
                for j in dec.outs:
                    if "aligned_angle" in data_d[dec][j]:
                        ang = data_d[dec][j]["aligned_angle"]
                        dt = get_D_matrix_lambda(ang, j.J, j.spins, j.spins)
                        aligned[j] = dt
                        idx = base_map[j]
                        ein = "{},{}->{}".format(idx_ein,"i"+idx+idx.upper(),idx_ein.replace(idx,idx.upper()))
                        amp_same = tf.einsum(ein, amp_same, dt)
            ret.append(amp_same)
        ret = tf.reduce_sum(ret, axis=0)
        return ret
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
            indices = indices.replace(base_map[i],'')
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
    a = Particle("A", J=1, P=-1, spins=(-1,1))
    b = Particle("B", J=1, P=-1)
    c = Particle("C", J=0, P=-1)
    d = Particle("D", J=1, P=-1)
    bd = Particle("BD", 1, 1)
    cd = Particle("CD", 1, 1)
    bc = Particle("BC", 1, 1)
    R = Particle("R", 1, 1)
    HelicityDecay(a, [bc, d])
    HelicityDecay(bc, [b, c])
    HelicityDecay(a, [cd, b])
    HelicityDecay(cd, [c, d])
    HelicityDecay(a, [bd, c])
    HelicityDecay(bd, [b, d])
    HelicityDecay(a, [R, c])
    HelicityDecay(R, [b, d])
    de = DecayGroup(a.chain_decay())
    data = prepare_data_from_decay(fnames, de)
    import time
    a= time.time()
    ret = de.get_amp(data)
    print(time.time() -a)
    return ret
