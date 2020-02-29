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
      warnings.warn("override breit wigner function :",name) # warning to users
    breit_wigner_dict[name_t] = f # function
    return f
  return fopt


@regist_lineshape("one")
def one(*args):
  return tf.complex(1.0,0.0) # breit_wigner_dict["one"]==tf.complex(1.0,0.0)


@regist_lineshape("BW")
def BW(m, m0,g0,*args):
  r"""
  .. math::
      BW(m) = \frac{1}{m_0^2 - m^2 -  i m_0 \Gamma_0 }
  
  """
  m0 = tf.cast(m0, m.dtype)
  gamma = tf.cast(g0, m.dtype)
  num = 1.0
  denom = tf.complex((m0 + m) * (m0 - m), -m0 * gamma)
  return num/denom


@regist_lineshape("default") # 两个名字
@regist_lineshape("BWR") # BW with running width
def BWR(m, m0,g0,q,q0,L,d):
  r"""
  .. math::
      BW(m) = \frac{1}{m_0^2 - m^2 -  i m_0 \Gamma(m)}
  
  """
  gamma = Gamma(m, g0, q, q0, L, m0, d)
  num = 1.0
  a = (tf.cast(m0, m.dtype) + m)
  b = (tf.cast(m0, m.dtype) - m)
  denom = tf.complex(a * b, -tf.cast(m0, m.dtype) * gamma)
  return num/denom

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

def Bprime_num(L,q,d):
  z = (q * d)**2
  return tf.sqrt(Bprime_polynomial(L, z))

def Bprime(L, q, q0, d):
  num = Bprime_num(L,q0,d)
  denom = Bprime_num(L,q,d)
  return tf.cast(num,denom.dtype)/denom

def barrier_factor(l,q,q0,d=3.0, axis=0): # cache q^l * B_l 只用于H里
  ret = []
  for i in l:
    tmp = q**i * tf.cast(Bprime(i,q,q0,d),q.dtype)
    ret.append(tmp)
  return tf.stack(ret)

def barrier_factor2(l,q,q0,d=3.0, axis=-1): # cache q^l * B_l 只用于H里
  ret = []
  for i in l:
    tmp = q**i * tf.cast(Bprime(i,q,q0,d),q.dtype)
    ret.append(tf.reshape(tmp, (-1, 1)))
  return tf.concat(ret, axis=axis)


def Bprime_polynomial(l, z):
    coeff = {
        0: [1.0],
        1: [1.0, 1.0],
        2: [1.0, 3.0, 9.0],
        3: [1.0, 6.0, 45.0, 225.0],
        4: [1.0, 10.0, 135.0, 1575.0, 11035.0],
        5: [1.0, 15.0, 315.0, 6300.0, 99225.0, 893025.0]
    }
    if l not in coeff:
        raise NotImplementedError
    z = tf.convert_to_tensor(z)
    cof = tf.convert_to_tensor(coeff[int(l+0.01)], dtype=z.dtype)
    return tf.math.polyval(cof, z)



