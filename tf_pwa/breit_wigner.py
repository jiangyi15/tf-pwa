import functools
import warnings
import tensorflow as tf

breit_wigner_dict = { }

def regist_lineshape(name=None):
  def fopt(f):
    name_t = name
    if name_t is None:
      name_t = f.__name__
    if name_t in breit_wigner_dict:
      warnings.warn("override breit wigner function :",name)
    breit_wigner_dict[name_t] = f
    return f
  return fopt

@regist_lineshape("one")
def one(*args):
  return tf.complex(1.0,0.0)

@regist_lineshape("BW")
def BW(m, m0,g0,*args):
  r"""
  .. math::
      BW(m) = \frac{1}{m_0^2 - m^2 -  i m_0 \Gamma_0 }
  
  """
  gamma = g0
  num = 1.0
  denom = tf.complex((m0 + m) * (m0 - m), -m0 * gamma)
  return num/denom

@regist_lineshape("default")
@regist_lineshape("BWR")
def BWR(m, m0,g0,q,q0,L,d):
  r"""
  .. math::
      BW(m) = \frac{1}{m_0^2 - m^2 -  i m_0 \Gamma(m)}
  
  """
  gamma = Gamma(m, g0, q, q0, L, m0, d)
  num = 1.0
  denom = tf.complex((m0 + m) * (m0 - m), -m0 * gamma)
  return num/denom

#@tf.function()
def Gamma(m, gamma0, q, q0, L, m0,d):
  r"""
  .. math::
      \Gamma(m) = \Gamma_0 \left(\frac{q}{q_0}\right)^{2L+1}\frac{m_0}{m} B_{L}'(q,q_0,d)
  
  """
  qq0 = (q / tf.cast(q0,q.dtype))**(2 * L + 1)
  mm0 = (tf.cast(m0,m.dtype) / m)
  bp = Bprime(L, q, q0, d)**2
  gammaM = gamma0 * qq0 * mm0 * tf.cast(bp,qq0.dtype)
  return gammaM


#@tf.function()
def Bprime_num(L,q,d):
  z = (q * d)**2
  if L == 0:
    return 1.0
  if L == 1:
    return tf.sqrt(1.0 + z)
  if L == 2:
    return tf.sqrt((9. + (3. + z) * z))
  if L == 3:
    return tf.sqrt( (z * (z * (z + 6.) + 45.) + 225.))
  if L == 4:
    return tf.sqrt((z * (z * (z * (z + 10.) + 135.) + 1575.) + 11025.));
  if L == 5:
    return tf.sqrt((z * (z * (z * (z * (z + 15.) + 315.) + 6300.) + 99225.) + 893025.));
  return 1.0

#@tf.function()
def Bprime(L, q, q0, d):
  num = Bprime_num(L,q0,d)
  denom = Bprime_num(L,q,d)
  return num/denom

#@tf.function()
def barrier_factor(l,q,q0,d=3.0):
  ret = []
  for i in l:
    tmp = q**i * tf.cast(Bprime(i,q,q0,d),q.dtype)
    #print(q**i,Bprime(i,q,q0,d))
    ret.append(tmp)
  return tf.stack(ret)
