import random
import numpy as np
from cg import cg_coef
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
  def __init__(self,name,mass,width,J,P):
    self.name = name
    self.mass = mass
    self.width = width
    self.J = J
    self.P = P

class Decay(object):
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
    return GetA2BC_LS_list(ja,jb,jc,pa,pb,pc)
  
  @functools.lru_cache()
  def get_l_list(self):
    return [ l for l,s in self.get_ls_list() ]
  
  def generate_params(self,ls=True):
    ret = []
    for l,s in self.get_ls_list():
      name_r = "{name}_ls_{l}_{s}_r".format(l,s)
      name_i = "{name}_ls_{l}_{s}_i".format(l,s)
      ret.append((name_r,name_i))
    return ret
  
  @functools.lru_cache()
  def get_cg_matrix(self):
    """
    [(l,s),(lambda_b,lambda_c)]
    cg_coef(jb, jc, lambda_b, -lambda_c, s, lambda_b - lambda_c) *
    cg_coef(l, s, 0, lambda_b - lambda_c, ja, lambda_b - lambda_c)
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
  
  def d_matrix(self,beta):
    j_total = self.mother.J
    
  def D_matrix(self,alpha,beta,gamma):
    return 0

def test():
  a = Particle("a",0.0,0.0,1,-1)
  b = Particle("b",0.0,0.0,1,-1)
  c = Particle("c",0.0,0.0,1,-1)
  decay = Decay(a,[b,c])
  print(decay.cg_matrix().T)
  print(np.array(decay.get_ls_list()))
  print(np.array(decay.get_ls_list())[:,0])

if __name__=="__main__":
  test()
  
