import tensorflow as tf
from math import pi
#from pysnooper import snoop
from .angle import LorentzVector
import numpy as np

def get_p(M, ma, mb):
  m2 = M * M
  m_p = (ma + mb)**2
  m_m = (ma - mb)**2
  p2 = (m2 - m_p) * (m2 - m_m)
  p = (p2 + tf.abs(p2)) / 2
  ret = tf.sqrt(p) / (2.0 * M)
  return ret

class PhaseSpaceGenerator(object):
  def __init__(self, m0, mass):
    self.m_mass = []
    self.setDecay(m0, mass)
    self.sum_mass = sum(self.m_mass)

  def generate_mass(self,n_iter):
    sm = self.sum_mass - self.m_mass[-1] - self.m_mass[-2]
    m_n  = self.m_mass[-1]
    ret = []
    for i in range(self.m_nt - 2):
      b = (self.m0 - sm)
      a = (m_n + self.m_mass[-i-2])
      random = tf.random.uniform([n_iter],dtype="float64")
      ms = (b-a) * random + a
      m_n = ms
      sm = sm - self.m_mass[-i-3]
      ret.append(ms)
    return ret
  
  def generate(self,n_iter,force=True,flatten=True):
    n_gen = 0
    n_total = n_iter
    mass = self.generate_mass(n_iter)
    if not flatten:
      pi = self.generate_momentum(mass)
      weight = self.get_weight(mass)
      return weight,pi
    mass_f = self.flatten_mass(mass)
    n_gen += mass_f[0].shape[0]
    while force and n_gen<n_iter:
      n_iter2 = int(1.01 * (n_total-n_gen)/(n_gen+1) * n_iter)
      n_iter2 = min(n_iter2,4000000)
      mass2 = self.generate_mass(n_iter2)
      mass_f2 = self.flatten_mass(mass2)
      n_gen += mass_f2[0].shape[0]
      n_total += n_iter2
      mass_f = [tf.concat([i,j],0) for i,j in zip(mass_f,mass_f2)]
    if force:
      mass_f = [i[:n_iter] for i in mass_f]
    return self.generate_momentum(mass_f)
  
  def generate_momentum(self,mass):
    n_iter = mass[0].shape[0]
    mass_t = [self.m_mass[-1]]
    for i in mass:
      mass_t.append(i)
    mass_t.append(self.m0)
    zeros = np.zeros([n_iter])
    p_list_0 = [ tf.stack([zeros + self.m_mass[-1], zeros, zeros, zeros], axis=-1)]
    # [ LorentzVector(zeros,zeros,zeros,self.m_mass[-1])]
    p_list = self.generate_momentum_i(mass_t[1],mass_t[0],-2,n_iter,p_list_0)
    #print("p_p2",p_list,p_list[0]+p_list[1])
    for i in range(1,self.m_nt-1):
      p_list = self.generate_momentum_i(mass_t[i+1],mass_t[i],-i-2,n_iter,p_list)
    #print("pi",p_list[0] ,(p_list[1]+p_list[2]).M())
    #print("sum",(p_list[0]+p_list[1]+p_list[2]).M())
    return p_list
  
  #@snoop()
  def generate_momentum_i(self,m0,m1,i,n_iter,p_list=[]):
    """
    |p| =  m0,m1,m2 in m0 rest frame
    """
    #print(i)
    m2 = self.m_mass[i]
    #print(m0,m1,m2)
    q = get_p(m0,m1,m2)
    cos_theta = 2*tf.random.uniform([n_iter],dtype="float64")-1
    sin_theta = tf.sqrt(1 - cos_theta*cos_theta)
    phi = 2*pi*tf.random.uniform([n_iter],dtype="float64")
    p_0 = tf.sqrt(q*q + m2*m2)
    #print(p_0)
    p_x = q*sin_theta*tf.cos(phi)
    p_y = q*sin_theta*tf.sin(phi)
    p_z = q*cos_theta
    p = tf.stack([p_0, p_x, p_y, p_z], axis=-1)
    # LorentzVector(p_x,p_y,p_z,p_0)
    ret = [p]
    p_boost = tf.stack([tf.sqrt(q*q + m1*m1), -p_x, -p_y, -p_z], axis=-1)
    # LorentzVector(-p.X,-p.Y,-p.Z,tf.sqrt(q*q + m1*m1))
    #print(p_boost.M())
    for i in p_list:
      ret.append(LorentzVector.rest_vector(-p_boost, i))
    return ret
  
  def flatten_mass(self,ms):
    weight = self.get_weight(ms)
    rnd = tf.random.uniform(weight.shape,dtype="float64")
    select = weight > rnd
    return [ tf.boolean_mask(i,select) for i in ms]
  
  def get_weight(self,ms):
    mass_t = [self.m_mass[-1]]
    for i in ms:
      mass_t.append(i)
    mass_t.append(self.m0)
    R = []
    for i in range(self.m_nt - 1):
      p = get_p(mass_t[i+1],mass_t[i],self.m_mass[-i-2])
      #print(p)
      R.append(p)
    wt = tf.math.reduce_prod(tf.stack(R),0)
    return wt/tf.cast(self.m_wtMax,wt.dtype)
  
  
  def setDecay(self, m0, mass):
    self.m0 = m0
    self.m_nt = len(mass)
    self.m_teCmTm = m0
    for i in range(self.m_nt):
      self.m_mass.append(mass[i])
      self.m_teCmTm -= mass[i]
    
    if self.m_teCmTm <= 0: return False 
    emmax = self.m_teCmTm + self.m_mass[-1]
    emmin = 0
    wtmax = 1
    for n in range(1,self.m_nt):
      emmin += self.m_mass[-n]
      emmax += self.m_mass[-n-1]
      p = get_p( emmax, emmin, self.m_mass[-n-1] )
      #print(p)
      wtmax *= p
    self.m_wtMax = wtmax
    return True
