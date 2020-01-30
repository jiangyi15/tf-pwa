import tensorflow as tf
import math
import numpy as np
import functools

def float2int(x): #utils里
  if x.dtype == np.int64 or x.dtype == np.int32:
    return x
  return np.floor(x+0.001).astype(np.int64)
  
class dfunctionJ(object):
  """
  wigner small d function for j
  """
  def __init__(self,j2,double=False):
    self.double = double # 是否翻倍成整数
    j2 = self.get_right_spin(j2) # 2*j
    self.j2 = j2
    self.m1 = list(range(-self.j2,self.j2+1,2))
    self.m2 = list(range(-self.j2,self.j2+1,2))
    self.w = tf.reshape(tf.constant(self.init_w(self.j2)),((self.j2+1)**2,self.j2+1)) # matrix of the prefactors
    self.sc = None # sin^λ*cos^μ
  
  def get_right_spin(self,x):
    ret = x
    if not self.double:
      if isinstance(ret,int):
        ret = 2*ret
      else :
        ret = float2int(ret*2)
    return ret
  
  @staticmethod
  def init_w(j): # the prefactor in the d-function of β
    r"""
    if  :math:`k \in [max(0,n-m),min(j-m,j+n)], l = 2k + m - n`
    
    .. math::
      
      w^{(j,m1,m2)}_{l} = (-1)^{k+m-n}\frac{\sqrt{(j+m)!(j-m)!(j+n)!(j-n)!}}{(j-m-k)!(j+n-k)!(k+m-n)!k!}
    
    else
    
    .. math::
      w^{(j,m1,m2)}_{l} = 0
    """
    ret = np.zeros(shape=(j+1,j+1,j+1))
    def f(x):
      return math.factorial(x>>1) #x>>1即x//2
    for m in range(-j,j+1,2):
      for n in range(-j,j+1,2):
        for k in range(max(0,n-m),min(j-m,j+n)+1,2): # d函数里求和的范围
          l = (2*k + (m - n))//2
          sign = (-1)**((k+m-n)//2) #和wiki一致
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
        raise Exception("need theta")
      else:
        return self.lazy_call(m1,m2) #没有theta就直接用self.sc
    
    a = tf.range(0,self.j2+1,1) #有theta就把lazy_init这块再算一遍
    sintheta = tf.sin(theta/2)
    costheta = tf.cos(theta/2)
    a = tf.reshape(a,(-1,1))
    a = tf.cast(tf.tile(a,[1,theta.shape[0]]),dtype=theta.dtype)
    s = tf.pow(tf.reshape(tf.tile(sintheta,[self.j2+1]),(self.j2+1,-1)),a)
    c = tf.pow(tf.reshape(tf.tile(costheta,[self.j2+1]),(self.j2+1,-1)),self.j2-a)
    self.sc = s*c
    
    m1 = self.get_right_spin(m1) #然后再lazy_call #可以改成直接调用
    m2 = self.get_right_spin(m2)
    tmpw = tf.gather(self.w,self.get_index(m1,m2))
    tmpw = tf.cast(tmpw,self.sc.dtype)
    sp = False
    if len(tmpw.shape) == 1:
      tmpw = tf.reshape(tmpw,(1,-1))
      sp = True
    ret = tf.matmul(tmpw ,self.sc)
    if sp:
      return tf.reshape(ret,self.sc.shape[1:])
    return ret
  
  def lazy_init(self,theta): # calculate sin^λ*cos^μ
    a = tf.range(0,self.j2+1)
    sintheta = tf.sin(theta/2)
    costheta = tf.cos(theta/2)
    a = tf.reshape(a,(-1,1))
    a = tf.cast(tf.tile(a,[1,theta.shape[0]]),dtype=theta.dtype)
    s = tf.pow(tf.reshape(tf.tile(sintheta,[self.j2+1]),(self.j2+1,-1)),a)
    c = tf.pow(tf.reshape(tf.tile(costheta,[self.j2+1]),(self.j2+1,-1)),self.j2-a)
    self.sc = s*c
    return self
  
  def get_index(self,m1,m2): #(m1,m2)整合成一位数组后的index
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


def ExpI_all(maxJ,phi): # matrix of cos(mφ)+isin(mφ)
  a = tf.range(-maxJ,maxJ+1,1.0)
  a = tf.reshape(a,(-1,1))
  phi = tf.reshape(phi,(1,-1)) # data #[1,2,3]变成[[1,2,3]]
  if not isinstance(phi,float):
    a = tf.cast(a,phi.dtype)
  mphi = tf.matmul(a,phi) #得到元素为m*phi的矩阵
  sinphi = tf.sin(mphi)
  cosphi = tf.cos(mphi)
  return tf.complex(cosphi,sinphi)

def Dfun_all(j,alpha,beta,gamma):
  d_fun = dfunctionJ(j)
  m = tf.range(-j,j+1)
  m1,m2=tf.meshgrid(m,m)
  d = d_fun(m2,m1,beta) # d-matrix
  expi_alpha = tf.reshape(ExpI_all(j,alpha),(2*j+1,1,-1))
  expi_gamma = tf.cast(tf.reshape(ExpI_all(j,gamma),(1,2*j+1,-1)),expi_alpha.dtype)
  dc = tf.complex(d,tf.zeros_like(d)) # cast to tf.complex
  return tf.cast(expi_alpha*expi_gamma,dc.dtype) * dc # D-matrix


def delta_D_trans(j,la,lb,lc):
  """
  (ja,ja) -> (ja,jb,jc)
  """
  s = np.zeros(shape=(len(la),len(lb),len(lc),(2*j+1),(2*j+1)))
  for i_a in range(len(la)):
    for i_b in range(len(lb)):
      for i_c in range(len(lc)):
        delta = lb[i_b]-lc[i_c]
        if abs(delta) <= j: # 物理要求
          s[i_a][i_b][i_c][la[i_a]+j][delta+j] = 1.0 #求和项里有的物理不允许则为0 #delta代表lb和lc的差
  return np.reshape(s,(len(la)*len(lb)*len(lc),(2*j+1)*(2*j+1)))
  
def Dfun_delta(ja,la,lb,lc,d):
  d = tf.reshape(d,((2*ja+1)*(2*ja+1),-1)) # d-function
  t = delta_D_trans(ja,la,lb,lc) # truth
  ret = tf.matmul(tf.cast(t,d.dtype),d) #作用是把d-matrix中非物理的去掉
  return tf.reshape(ret,(len(la),len(lb),len(lc),-1))


class D_Cache(object): 
  def __init__(self,alpha,beta,gamma=0.0):
    self.alpha = alpha
    self.gamma = gamma
    self.beta = beta
    self.cachej = {}
    
  @functools.lru_cache()
  def __call__(self,j,m1=None,m2=None): # cached D-matrix
    if j not in self.cachej:
      self.cachej[j] = Dfun_all(j,self.alpha,self.beta,self.gamma)
    if m1 is None:
      return self.cachej[j]
    else :
      return self.cachej[m1+j][m2+j]
    
  @functools.lru_cache()
  def _tuple_get_lambda(self,j,la,lb,lc=(0,)): # λ才需要用Dfun_delta的物理约束
    d = self(j)
    return Dfun_delta(j,la,lb,lc,d)
  def get_lambda(self,j,la,lb,lc=[0]): #lc缺省为0则delta就是lb
    return self._tuple_get_lambda(j,tuple(la),tuple(lb),tuple(lc))


if __name__ == "__main__":
  a = dfunctionJ(3)
  print(a(np.array([[2,-2],[-2,2]]),np.array([2,-2]),np.array([0.0,1.0,2.0,3.0])))
