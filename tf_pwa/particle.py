import functools
import numpy as np
# from pysnooper import snoop

from .cg import cg_coef
from .breit_wigner import barrier_factor as default_barrier_factor
from .dec_parser import load_dec_file
from .utils import deep_ordered_iter


def cross_combine(x):
  if not x: # if x is []
    return []
  head = x[0]
  tail = x[1:]
  ret = []
  other = cross_combine(tail)
  for i in head:
    if not other:
      ret.append(i)
    else:
      for j in other:
        ret.append(i + j)
  return ret

class BaseParticle(object):
  """
  Base Particle object
  """
  def __init__(self, name, id_=None):
    """
    name is "name[:id]"
    """
    if id_ is None:
      names = name.split(":")
      if len(names) > 1:
        self.name = ":".join(names[:-1])
        self._id = int(names[-1])
      else:
        self.name, self._id = name, 0
    else:
      self.name, self._id = name, id_
    self.decay = [] # list of Decay
    self.creators = [] # list of Decay which creates the particle

  def add_decay(self, d):
    if d not in self.decay:
      self.decay.append(d)

  def remove_decay(self, d):
    self.decay.remove(d)

  def add_creator(self, d):
    self.creators.append(d)

  def __repr__(self):
    if self._id == 0:
      return self.name
    return "{}:{}".format(self.name, self._id)
  def __hash__(self):
    return hash((self.name, self._id))
  def __eq__(self, other):
    if not isinstance(other, BaseParticle):
      return False
    return (self.name, self._id) == (other.name, other._id)
  def __lt__(self, other):
    if isinstance(other, BaseParticle):
      return (self.name, self._id) < (other.name, other._id)
    return self.name < other
  def __gt__(self, other):
    if isinstance(other, BaseParticle):
      return (self.name, self._id) > (other.name, other._id)
    return self.name > other

  def chain_decay(self):
    ret = []
    for i in self.decay:
      ret_tmp = [[[i]]]
      for j in i.outs:
        tmp = j.chain_decay()
        if tmp: # if tmp is not []
          ret_tmp.append(tmp)
      ret += cross_combine(ret_tmp)
    return ret #最后出来个啥？

  def get_resonances(self):
    decay_chain = self.chain_decay()
    chains = [DecayChain(i) for i in decay_chain]
    decaygroup = DecayGroup(chains)
    return decaygroup.resonances

class Particle(BaseParticle): # add parameters to BaseParticle
    """
    general Particle object
    """
    def __init__(self, name, J=0, P=-1, spins=None, mass=None, width=None):
        super(Particle, self).__init__(name)
        self.J = J
        self.P = P
        if spins is None:
            spins = tuple(range(-J, J+1))
        self.spins = tuple(spins)
        self.mass = mass
        self.width = width

def GetA2BC_LS_list(ja, jb, jc, pa, pb, pc):
  dl = 0 if pa * pb * pc == 1 else  1 # pa = pb * pc * (-1)^l
  s_min = abs(jb - jc)
  s_max = jb + jc
  # ns = s_max - s_min + 1
  ret = []
  for s in range(s_min, s_max+1):
    for l in range(abs(ja - s), ja + s + 1):
      if l % 2 == dl:
        ret.append((l, s))
  return ret

def simple_cache_fun(f):
  name = "simple_cached_"+f.__name__
  @functools.wraps(f)
  def g(self):
    if not hasattr(self,name):
      setattr(self, name, f(self))
    return getattr(self, name)
  return g


class BaseDecay(object):
  """
  Base Decay object
  """
  def __init__(self, core, outs, name=None, disable=False):
    self.name = name
    self.core = core # mother particle
    self.outs = tuple(outs) # daughter particles
    if not disable:
      self.core.add_decay(self)
      for i in outs:
        i.add_creator(self)

  def __repr__(self):
    ret = str(self.core)
    ret += "->"
    ret += "+".join([str(i) for i in self.outs])
    return ret # "A->B+C"

  @simple_cache_fun#@functools.lru_cache()
  def get_id(self):
    return (self.core, tuple(sorted(self.outs)))

  def __hash__(self):
    return hash(self.get_id())

  def __eq__(self, other):
    if not isinstance(other, BaseDecay):
      return False
    return self.get_id() == other.get_id()

  def __lt__(self, other):
    if not isinstance(other, BaseDecay):
      return False
    return self.get_id() < other.get_id()


class Decay(BaseDecay): # add useful methods to BaseDecay
  """
  general Decay object
  """
  @functools.lru_cache()
  def get_ls_list(self):
    ja = self.core.J
    jb = self.outs[0].J
    jc = self.outs[1].J
    pa = self.core.P
    pb = self.outs[0].P
    pc = self.outs[1].P
    return tuple(GetA2BC_LS_list(ja, jb, jc, pa, pb, pc))

  @functools.lru_cache()
  def get_l_list(self):
    return tuple([l for l, s in self.get_ls_list()])

  @functools.lru_cache()
  def get_min_l(self):
    return min(self.get_l_list())

  def generate_params(self, name=None, _ls=True): # 取代amplitude里的gen_coef？
    if name is None:
      name = self.name
    ret = []
    for l, s in self.get_ls_list():
      name_r = "{name}_l{l}_s{s}_r".format(name=name, l=l, s=s)
      name_i = "{name}_l{l}_s{s}_i".format(name=name, l=l, s=s)
      ret.append((name_r, name_i))
    return ret

  @functools.lru_cache()
  def get_cg_matrix(self): # CG factor inside H
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
    n = (2*jb + 1)*(2*jc + 1)
    ret = np.zeros(shape=(n, m))
    for i, ls_i in enumerate(ls):
      l, s = ls_i
      j = 0
      for lambda_b in range(-jb, jb+1):
        for lambda_c in range(-jc, jc+1):
          ret[j][i] = np.sqrt((2 * l + 1) / (2 * ja + 1)) \
                      * cg_coef(jb, jc, lambda_b, -lambda_c, s, lambda_b - lambda_c) \
                      * cg_coef(l, s, 0, lambda_b - lambda_c, ja, lambda_b - lambda_c)
          j += 1
    return ret

  def barrier_factor(self, q, q0): # Barrier factor inside H
    """
    defalut_barrier_factor
    """
    d = 3.0
    ret = default_barrier_factor(self.get_l_list(), q, q0, d)
    return ret


def split_particle_type(decays):
    core_particles = set()
    out_particles = set()

    for i in decays:
      core_particles.add(i.core)
      for j in i.outs:
        out_particles.add(j)

    inner = core_particles & out_particles
    top = core_particles - inner
    outs = out_particles - inner
    return top, inner, outs # top, intermediate, outs particles

def split_len(dicts):
    """
    {"b":[1],"c":[1,2],"d":[2]} => [None,{"b":[1],"d":[2]},{"c":[1,2]}]
    """
    size_table = []
    for i in dicts:
        tmp = dicts[i]
        size_table.append((len(tmp), i))
    max_l = max([i for i, _ in size_table])
    ret = [None] * (max_l+1)
    for i, s in size_table:
        if ret[i] is None:
            ret[i] = []
        ret[i].append((s, dicts[s]))
    return ret

class DecayChain(object):
  def __init__(self, chain):
    self.chain = chain
    top, self.inner, outs = split_particle_type(chain)
    assert len(top) == 1, "top particles must be only one particle"
    self.top = top.pop()
    self.outs = sorted(list(outs))

  def __iter__(self):
    return iter(self.chain)

  def __repr__(self):
    return "{}".format(self.chain)

  @functools.lru_cache()
  def sorted_table(self):
    """
    A topology independent structure
    [a->rb,r->cd] => {a:[b,c,d],r:[c,d],b:[b],c:[c],d:[d]}
    """
    decay_dict = {}
    for i in self.outs:
      decay_dict[i] = [i]

    chain = self.chain
    while chain:
      tmp_chain = []
      for i in chain:
        if all([j in decay_dict for j in i.outs]):
          decay_dict[i.core] = []
          for j in i.outs:
            decay_dict[i.core] += decay_dict[j]
          decay_dict[i.core].sort()
        else:
          tmp_chain.append(i)
      chain = tmp_chain
    decay_dict[self.top] = sorted(list(self.outs))
    return decay_dict

  def sorted_table_layers(self):
    st = self.sorted_table()
    return split_len(st)

  @staticmethod
  def from_sorted_table(decay_dict):
    """
    Create decay chain form a topology independent structure
    {a:[b,c,d],r:[c,d],b:[b],c:[c],d:[d]} => [a->rb,r->cd]
    """
    def sum_list(ls):
      ret = ls[0]
      for i in ls[1:]:
        ret = ret + i
      return ret

    def deep_search(idx, base):
      base_step = 2
      max_step = len(base)
      while base_step <= max_step:
        for i in deep_ordered_iter(base, base_step):
          check = sum_list([base[j] for j in i])
          if sorted(check) == sorted(idx[1]):
            return i
        base_step += 1

    s_dict = split_len(decay_dict)
    base_dict = dict(s_dict[1])
    ret = []
    for s_dict_i in s_dict[2:]:
      if s_dict_i:
        for j in s_dict_i:
          found = deep_search(j, base_dict)
          ret.append(BaseDecay(j[0], found, disable=True))
          for i in found:
            del base_dict[i]
          base_dict[j[0]] = j[1]
    return DecayChain(ret)

  @staticmethod
  def from_particles(top, finals):
    """
    build possible Decay Chain Topology
    a -> [b,c,d] => [[a->rb,r->cd],[a->rc,r->bd],[a->rd,r->bc]]
    """
    assert len(finals) > 0, " "

    def get_graphs(g, ps):
      if ps:
        p = ps[0]
        ps = ps[1:]
        ret = []
        for i in g.edges:
          gi = g.copy()
          gi.add_node(i, p)
          ret += get_graphs(gi, ps)
        return ret
      return [g]
    base = _Chain_Graph()
    base.add_edge(top, finals[0])
    gs = get_graphs(base, finals[1:])
    return [gi.get_decay_chain(top, head="chain{}_".format(i)) for i, gi in enumerate(gs)]

  @functools.lru_cache()
  def topology_id(self, identical=True):
    a = self.sorted_table()
    if identical:
      set_a = [[j.name for j in a[i]] for i in a]
    else:
      set_a = [list(a[i]) for i in a]
    return sorted(set_a)

  def topology_map(self, other):
    """
    [A->R+B,R->C+D],[A->Z+B,Z->C+D] => {A:A,B:B,C:C,D:D,R:Z,A->R+B:A->Z+B,R->C+D:Z->C+D}
    """
    a = self.sorted_table()
    b = other.sorted_table()
    ret = {}
    for i in a:
      for j in b:
        if a[i] == b[j]:
          ret[i] = j
          break
    for i in self:
      test_decay = BaseDecay(ret[i.core], [ret[k] for k in i.outs], disable=False)
      for j in other:
        if test_decay == j:
          ret[i] = j
          break
    return ret

  def topology_same(self, other, identical=True):
    if not isinstance(other, DecayChain):
      raise TypeError("unsupport type {}".format(type(other)))
    return self.topology_id(identical) == other.topology_id(identical)


class _Chain_Graph(object):
  def __init__(self):
    self.nodes = []
    self.edges = []
    self.count = 0
  def add_edge(self, a, b):
    self.edges.append((a, b))
  def add_node(self, e, d):
    self.edges.remove(e)
    count = self.count
    node = "node_{}".format(count)
    self.nodes.append(node)
    self.edges.append((e[0], node))
    self.edges.append((node, e[1]))
    self.edges.append((node, d))
    self.count += 1
  def copy(self):
    ret = _Chain_Graph()
    ret.nodes = self.nodes.copy()
    ret.edges = self.edges.copy()
    ret.count = self.count
    return ret
  def get_decay_chain(self, top, head="tmp_"):
    decay_list = {}
    ret = []
    inner_particle = {}
    for i in self.nodes:
      inner_particle[i] = BaseParticle("{}{}".format(head, i))
    for i, j in self.edges:
      i = inner_particle.get(i, i)
      j = inner_particle.get(j, j)
      if i in decay_list:
        decay_list[i].append(j)
      else:
        decay_list[i] = [j]
    assert len(decay_list[top]) == 1, ""
    tmp = decay_list[top][0]
    decay_list[top] = decay_list[tmp]
    del decay_list[tmp]
    for i in decay_list:
      tmp = BaseDecay(i, decay_list[i], disable=True)
      ret.append(tmp)
    return DecayChain(ret)

class DecayGroup(object):
  def __init__(self, chains):
    first_chain = chains[0]
    if not isinstance(first_chain, DecayChain):
      chains = [DecayChain(i) for i in chains]
      first_chain = chains[0]
    self.chains = chains
    self.top = first_chain.top
    self.outs = sorted(list(first_chain.outs))
    for i in chains:
      assert i.top == first_chain.top, ""
      assert i.outs == first_chain.outs, ""
    resonances = set()
    for i in chains:
      resonances |= i.inner
    self.resonances = list(resonances)

  def __repr__(self):
    return "{}".format(self.chains)

  def __iter__(self):
    return iter(self.chains)

  def topology_structure(self, identical=False):
    ret = []
    for i in self:
      for j in ret:
        if i.topology_same(j, identical):
          break
      else:
        ret.append(i)
    return ret
  
  @functools.lru_cache()
  def get_chains_map(self):
    chain_maps = []
    for decays in self.topology_structure():
        decay_chain = DecayChain(list(decays))
        tmp = {}
        for j in self:
            if decay_chain.topology_same(j):
                chain_map = decay_chain.topology_map(j)
                tmp[j] = chain_map
        chain_maps.append(tmp)
    return chain_maps


def load_decfile_particle(fname):
  with open(fname) as f:
    dec = load_dec_file(f)
  dec = list(dec)
  particles = {}
  def get_particles(name):
    if not name in particles:
      a = Particle(name)
      particles[name] = a
    return particles[name]
  decay = []
  for i in dec:
    cmd, var = i
    if cmd == "Particle":
      a = get_particles(var["name"])
      setattr(a, "params", var["params"])
    if cmd == "Decay":
      for j in var["final"]:
        outs = [get_particles(k) for k in j["outs"]]
        de = Decay(get_particles(var["name"]), outs)
        for k in j:
          if k != "outs":
            setattr(de, k, j[k])
        decay.append(de)
    if cmd == "RUNNINGWIDTH":
      pa = get_particles(var[0])
      setattr(pa, "running_width", True)
  top, inner, outs = split_particle_type(decay)
  return top, inner, outs


    

def test():
  a = Particle("a", 1, -1)
  b = Particle("b", 1, -1)
  c = Particle("c", 0, -1)
  d = Particle("d", 1, -1)
  tmp = Particle("tmp", 1, -1)
  tmp2 = Particle("tmp2", 1, -1)
  decay = Decay(a, [tmp, c])
  decay2 = Decay(tmp, [b, d])
  decay3 = Decay(a, [tmp2, d])
  decay4 = Decay(tmp2, [b, c])
  decaychain = DecayChain([decay, decay2])
  decaychain2 = DecayChain([decay3, decay4])
  decaygroup = DecayGroup([decaychain, decaychain2])
  print(decay.get_cg_matrix().T)
  print(np.array(decay.get_ls_list()))
  print(np.array(decay.get_ls_list())[:, 0])
  print(decaychain)
  print(decaychain.sorted_table())
  print(decaygroup)
  print(a.get_resonances())

if __name__ == "__main__":
  test()
