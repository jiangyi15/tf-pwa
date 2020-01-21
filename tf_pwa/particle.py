import random
import numpy as np
from .cg import cg_coef
from .breit_wigner import barrier_factor as default_barrier_factor
import functools

def GetA2BC_LS_list(ja,jb,jc,pa,pb,pc):
  dl = 0 if pa * pb * pc == 1 else  1 # pa = pb * pc * (-1)^l
  s_min = abs(jb - jc);
  s_max = jb + jc;
  ns = s_max - s_min + 1
  ret = []
  for s in range(s_min,s_max+1):
    for l in range(abs(ja - s),ja + s +1 ):
      if l % 2 == dl :
        ret.append((l,s))
  return ret


class Particle(object):
  """
  general Particle object
  """
  def __init__(self,name,J,P,spins=None):
    self.name = name
    self.J = J
    self.P = P
    self.decay = []
    if spins is None:
      spins = list(range(-J,J+1))
    self.spins = spins
  
  def add_decay(self,d):
    self.decay.append(d)
  
  def remove_decay(self,d):
    self.decay.remove(d)
  
  def __repr__(self):
    return self.name

  def chain_decay(self):
    ret = []
    for i in self.decay:
      ret_tmp = [[[i]]]
      for j in i.outs:
        tmp = j.chain_decay()
        if len(tmp)>0:
          ret_tmp.append(tmp)
      ret += cross_combine(ret_tmp)
    return ret
  
  def get_resonances(self):
    decay_chain = self.chain_decay()
    chains = [DecayChain(i) for i in decay_chain]
    decaygroup = DecayGroups(chains)
    return decaygroup.resonances
    
def cross_combine(x):
  if len(x)==0:
    return []
  head = x[0]
  tail = x[1:]
  ret = []
  other = cross_combine(tail)
  for i in head:
    if len(other) == 0:
      ret.append(i)
    else:
      for j in other:
        ret.append(i+j)
  return ret
        

class Decay(object):
  """
  general Decay object
  """
  def __init__(self,core,outs,name=None):
    self.name = name
    self.core = core
    self.core.add_decay(self)
    self.outs = outs
  
  @functools.lru_cache()
  def get_ls_list(self):
    ja = self.core.J
    jb = self.outs[0].J
    jc = self.outs[1].J
    pa = self.core.P
    pb = self.outs[0].P
    pc = self.outs[1].P
    return tuple(GetA2BC_LS_list(ja,jb,jc,pa,pb,pc))
  
  @functools.lru_cache()
  def get_l_list(self):
    return tuple([ l for l,s in self.get_ls_list() ])
  
  @functools.lru_cache()
  def get_min_l(self):
    return min(self.get_l_list())
  
  def generate_params(self,name=None,ls=True):
    if name is None:
      name = self.name
    ret = []
    for l,s in self.get_ls_list():
      name_r = "{name}_l{l}_s{s}_r".format(name=name,l=l,s=s)
      name_i = "{name}_l{l}_s{s}_i".format(name=name,l=l,s=s)
      ret.append((name_r,name_i))
    return ret
  
  @functools.lru_cache()
  def get_cg_matrix(self):
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
    n = (2*jb + 1)*(2*jc+1)
    ret = np.zeros(shape=(n,m))
    for i,ls_i in enumerate(ls):
      l,s = ls_i
      j = 0
      for lambda_b in range(-jb,jb+1):
        for lambda_c in range(-jc,jc+1):
          ret[j][i] = np.sqrt((2*l+1)/(2*ja+1)) \
                      * cg_coef(jb, jc, lambda_b, -lambda_c, s, lambda_b - lambda_c) \
                      * cg_coef(l, s, 0, lambda_b - lambda_c, ja, lambda_b - lambda_c)
          j += 1
    return ret
  
  def barrier_factor(self,q,q0):
    """
    defalut_barrier_factor
    """
    d = 3.0
    ret = default_barrier_factor(self.get_l_list(),q,q0,d)
    return ret
  
  def __repr__(self):
    ret = str(self.core)
    ret += "->"
    ret += str(self.outs[0])
    for i in self.outs[1:]:
      ret += "+"+str(i)
    return ret
    
class DecayChain(object):
  def __init__(self,chain):
    self.chain = chain
    self.top, self.inner, self.outs = self.determine_top_particle(chain)
    
  @staticmethod
  def determine_top_particle(chain):
    core_particles = set()
    out_particles = set()
    
    for i in chain:
      core_particles.add(i.core)
      for j in i.outs:
        out_particles.add(j)
    
    inner = core_particles & out_particles
    top = core_particles - inner
    outs = out_particles - inner
    return top, inner, outs
  
  def __repr__(self):
    return "{}".format(self.chain)

class DecayGroups(object):
  def __init__(self,chains):
    first_chain = chains[0]
    if not isinstance(first_chain,DecayChain):
      chains = [DecayChain(i) for i in chains]
      first_chain = chains[0]
    self.chains = chains
    assert len(first_chain.top) == 1,"top particles must be only one particle"
    self.top = list(first_chain.top)[0]
    self.outs = list(first_chain.outs)
    for i in chains:
      assert i.top == first_chain.top,""
      assert i.outs == first_chain.outs,""
    resonances = set()
    for i in chains:
      resonances |= i.inner
    self.resonances = list(resonances)
    
  def __repr__(self):
    return "{}".format(self.chains)
    
  

def test():
  a = Particle("a",1,-1)
  b = Particle("b",1,-1)
  c = Particle("c",0,-1)
  d = Particle("d",1,-1)
  tmp = Particle("tmp",1,-1)
  tmp2 = Particle("tmp2",1,-1)
  decay = Decay(a,[tmp,c])
  decay2 = Decay(tmp,[b,d])
  decay3 = Decay(a,[tmp2,d])
  decay4 = Decay(tmp2,[b,c])
  decaychain = DecayChain([decay,decay2])
  decaychain2 = DecayChain([decay3,decay4])
  decaygroup = DecayGroups([decaychain,decaychain2])
  print(decay.get_cg_matrix().T)
  print(np.array(decay.get_ls_list()))
  print(np.array(decay.get_ls_list())[:,0])
  print(decaychain)
  print(decaygroup)
  print(a.get_resonances())

if __name__=="__main__":
  test()
  
