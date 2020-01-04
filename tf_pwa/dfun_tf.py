import tensorflow as tf
import math
import numpy as np
import functools

def float2int(x):
  if x.dtype == np.int64 or x.dtype == np.int32:
    return x
  return np.floor(x+0.001).astype(np.int64)
  
class dfunctionJ(object):
  """
  wigner small d function for j
  """
  def __init__(self,j2,double=False):
    self.double = double
    j2 = self.get_right_spin(j2)
    self.j2 = j2
    self.m1 = list(range(-self.j2,self.j2+1,2))
    self.m2 = list(range(-self.j2,self.j2+1,2))
    self.w = tf.reshape(tf.constant(self.init_w(self.j2)),((self.j2+1)**2,self.j2+1))
    self.sc = None
  
  def get_right_spin(self,x):
    ret = x
    if not self.double:
      if isinstance(ret,int):
        ret = 2*ret
      else :
        ret = float2int(ret*2)
    return ret
  
  @staticmethod
  def init_w(j):
    """
    if 
      k \in [max(0,n-m),min(j-m,j+n)], l = 2k + m - n
      w^{(j,m1,m2)}_{l} = (-1)^(k+m-n)\frac{\sqrt{(j+m)!(j-m)!(j+n)!(j-n)!}}{(j-m-k)!(j+n-k)!(k+m-n)!k!}
    else 
      w^{(j,m1,m2)}_{l} = 0
    """
    ret = np.zeros(shape=(j+1,j+1,j+1))
    def f(x):
      return math.factorial(x>>1)
    for m in range(-j,j+1,2):
      for n in range(-j,j+1,2):
        for k in range(max(0,n-m),min(j-m,j+n)+1,2):
          l = (2*k + (m - n))//2
          sign = (-1)**((k+m-n)//2)
          tmp = sign * math.sqrt(1.0*f(j+m)*f(j-m)*f(j+n)*f(j-n))
          tmp /= f(j-m-k)*f(j+n-k)*f(k+m-n)*f(k)
          ret[(m+j)//2][(n+j)//2][l] = tmp
    return ret
  
  def __call__(self,m1,m2,theta=None):
    """
    d^{j}_{m1,m2}(\theta) = \sum_{l=0}^{2j} w_{l}^{(j,m1,m2)} sin(\theta/2)^{l} cos(\theta/2)^{2j-l}
    """
    
    if theta is None:
      if self.sc is None:
        raise "need theta"
      else:
        return self.lazy_call(m1,m2)
    m1 = self.get_right_spin(m1)
    m2 = self.get_right_spin(m2)
    a = tf.range(0,self.j2+1,1)
    sintheta = tf.sin(theta/2)
    costheta = tf.cos(theta/2)
    a = tf.reshape(a,(-1,1))
    a = tf.cast(tf.tile(a,[1,theta.shape[0]]),dtype=theta.dtype)
    s = tf.pow(tf.reshape(tf.tile(sintheta,[self.j2+1]),(self.j2+1,-1)),a)
    c = tf.pow(tf.reshape(tf.tile(costheta,[self.j2+1]),(self.j2+1,-1)),self.j2-a)
    self.sc = s*c
    sp = False
    tmpw = tf.gather(self.w,self.get_index(m1,m2))
    tmpw = tf.cast(tmpw,self.sc.dtype)
    if len(tmpw.shape) == 1:
      tmpw = tf.reshape(tmpw,(1,-1))
      sp = True
    ret = tf.matmul(tmpw ,self.sc)
    if sp:
      return tf.reshape(ret,self.sc.shape[1:])
    return ret
  
  def lazy_init(self,theta):
    a = tf.range(0,self.j2+1)
    sintheta = tf.sin(theta/2)
    costheta = tf.cos(theta/2)
    a = tf.reshape(a,(-1,1))
    a = tf.cast(tf.tile(a,[1,theta.shape[0]]),dtype=theta.dtype)
    s = tf.pow(tf.reshape(tf.tile(sintheta,[self.j2+1]),(self.j2+1,-1)),a)
    c = tf.pow(tf.reshape(tf.tile(costheta,[self.j2+1]),(self.j2+1,-1)),self.j2-a)
    self.sc = s*c
    return self
  
  def get_index(self,m1,m2):
    i_m1 = (m1+self.j2)//2
    i_m2 = (m2+self.j2)//2
    return i_m1 * (self.j2+1) + i_m2

  def lazy_call(self,m1,m2):
    m1 = self.get_right_spin(m1)
    m2 = self.get_right_spin(m2)
    tmpw = tf.gather(self.w,self.get_index(m1,m2))
    tmpw = tf.cast(tmpw,self.sc.dtype)
    sp = False
    if len(tmpw.shape) == 1:
      tmpw = tf.reshape(tmpw,(1,-1))
      sp = True
    ret = tf.matmul(tmpw ,self.sc)
    if sp:
      return tf.constant(tf.reshape(ret,self.sc.shape[1:]))
    return tf.constant(ret)

class ExpI_Cache(object):
  def __init__(self,phi,maxJ = 2):
    self.maxj = maxJ
    self.phi = phi
    a = tf.range(-maxJ,maxJ+1,1.0)
    a = tf.reshape(a,(-1,1))
    phi = tf.reshape(phi,(1,-1))
    mphi = tf.matmul(a,phi)
    self.sinphi = tf.sin(mphi)
    self.cosphi = tf.cos(mphi)
  def __call__(self,m):
    idx = m + self.maxj
    return complex(self.cosphi[idx],self.sinphi[idx])

class D_fun_Cache(object):
  def __init__(self,alpha,beta,gamma=0.0):
    self.alpha = ExpI_Cache(alpha)
    self.gamma = ExpI_Cache(gamma)
    self.beta = beta
    self.dfuncj = {}
  @functools.lru_cache()
  def __call__(self,j,m1=None,m2=None):
    if abs(m1) > j or abs(m2) > j:
      return 0.0
    if j not in self.dfuncj:
      self.dfuncj[j] = dfunctionJ(j)
      self.dfuncj[j].lazy_init(self.beta)
    d = self.dfuncj[j](m1,m2)
    return self.alpha(m1)*self.gamma(m2)*d

def Dfun_cos(j,m1,m2,alpha,cosbeta,gamma):
  tmp = complex(0.0,alpha * m1 + gamma * m2).exp() * dfunction(j, m1, m2, cosbeta)
  return tmp

def ExpI_all(maxJ,phi):
  a = tf.range(-maxJ,maxJ+1,1.0)
  a = tf.reshape(a,(-1,1))
  phi = tf.reshape(phi,(1,-1))
  if not isinstance(phi,float):
    a = tf.cast(a,phi.dtype)
  mphi = tf.matmul(a,phi)
  sinphi = tf.sin(mphi)
  cosphi = tf.cos(mphi)
  return tf.complex(cosphi,sinphi)

def Dfun_all(j,alpha,beta,gamma):
  d_fun = dfunctionJ(j)
  m = tf.range(-j,j+1)
  m1,m2=tf.meshgrid(m,m)
  d = d_fun(m2,m1,beta)
  expi_alpha = tf.reshape(ExpI_all(j,alpha),(2*j+1,1,-1))
  expi_gamma = tf.cast(tf.reshape(ExpI_all(j,gamma),(1,2*j+1,-1)),expi_alpha.dtype)
  #a = tf.tile(expi_alpha,[1,2*j+1,1])
  #b = tf.tile(expi_gamma,[2*j+1,1,1])
  dc = tf.complex(d,tf.zeros_like(d))
  return tf.cast(expi_alpha*expi_gamma,dc.dtype) * dc

def delta_D_trans(j,la,lb,lc):
  """
  (ja,ja) -> (ja,jb,jc)
  """
  s = np.zeros(shape=(len(la),len(lb),len(lc),(2*j+1),(2*j+1)))
  for i_a in range(len(la)):
    for i_b in range(len(lb)):
      for i_c in range(len(lc)):
        delta = lb[i_b]-lc[i_c]
        if abs(delta) <= j:
          s[i_a][i_b][i_c][la[i_a]+j][delta+j] = 1.0
  return np.reshape(s,(len(la)*len(lb)*len(lc),(2*j+1)*(2*j+1)))
  

def Dfun_delta(ja,la,lb,lc,d):
  d = tf.reshape(d,((2*ja+1)*(2*ja+1),-1))
  t = delta_D_trans(ja,la,lb,lc)
  ret = tf.matmul(tf.cast(t,d.dtype),d)
  return tf.reshape(ret,(len(la),len(lb),len(lc),-1))

class D_Cache(object):
  def __init__(self,alpha,beta,gamma=0.0):
    self.alpha = alpha
    self.gamma = gamma
    self.beta = beta
    self.cachej = {}
  @functools.lru_cache()
  def __call__(self,j,m1=None,m2=None):
    if j not in self.cachej:
      self.cachej[j] = Dfun_all(j,self.alpha,self.beta,self.gamma)
    if m1 is None:
      return self.cachej[j]
    else :
      return self.cachej[m1+j][m2+j]

  def get_lambda(self,j,la,lb,lc=[0]):
    d = self(j)
    return Dfun_delta(j,la,lb,lc,d)

if __name__ == "__main__":
  a = dfunctionJ(3)
  print(a(np.array([[2,-2],[-2,2]]),np.array([2,-2]),np.array([0.0,1.0,2.0,3.0])))
