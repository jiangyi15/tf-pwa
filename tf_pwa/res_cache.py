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
    if spins is None:
      spins = list(range(-J,J+1))
    self.spins = spins
  
  def __repr__(self):
    return self.name

class Decay(object):
  """
  general Decay object
  """
  def __init__(self,name,mother,outs):
    self.name = name
    self.mother = mother
    self.outs = outs
  
  @functools.lru_cache()
  def get_ls_list(self):
    ja = self.mother.J
    jb = self.outs[0].J
    jc = self.outs[1].J
    pa = self.mother.P
    pb = self.outs[0].P
    pc = self.outs[1].P
    return tuple(GetA2BC_LS_list(ja,jb,jc,pa,pb,pc))
  
  @functools.lru_cache()
  def get_l_list(self):
    return tuple([ l for l,s in self.get_ls_list() ])
  
  @functools.lru_cache()
  def get_min_l(self):
    return min(self.get_l_list())
  
  def generate_params(self,ls=True):
    ret = []
    for l,s in self.get_ls_list():
      name_r = "{name}_l{l}_s{s}_r".format(l,s)
      name_i = "{name}_l{l}_s{s}_i".format(l,s)
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
    ja = self.mother.J
    jb = self.outs[0].J
    jc = self.outs[1].J
    n = (2*jb + 1)*(2*jc+1)
    ret = np.zeros(shape=(n,m))
    for i in range(len(ls)):
      l,s = ls[i]
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
    ret = str(self.mother)
    ret += "->"
    ret += str(self.outs[0])
    for i in self.outs[1:]:
      ret += "+"+str(i)
    return ret
    
  

def test():
  a = Particle("a",1,-1)
  b = Particle("b",1,-1)
  c = Particle("c",1,-1)
  decay = Decay(a,[b,c])
  print(decay.cg_matrix().T)
  print(np.array(decay.get_ls_list()))
  print(np.array(decay.get_ls_list())[:,0])

if __name__=="__main__":
  test()
  
