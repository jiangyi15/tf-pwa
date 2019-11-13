import tensorflow as tf
import math
import numpy as np
import functools

def float2int(x):
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

if __name__ == "__main__":
  a = dfunctionJ(3)
  print(a(np.array([[2,-2],[-2,2]]),np.array([2,-2]),np.array([0.0,1.0,2.0,3.0])))
