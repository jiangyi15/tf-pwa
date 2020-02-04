import numpy as np

def bounds_trans(x,a=None,b=None):
  """
  x \in R -> x \in (a,b)
  """
  if a is None:
    if b is None:
      # (-inf,inf)
      return x
    else:
      # (-inf,b)
      return b+1-np.sqrt(x**2+1)
  else :
    if b is None:
      # (a,inf)
      return a-1+np.sqrt(x**2+1)
    else:
      return (b-a)/2*(np.sin(x)+1) + a

def trans_non(x):#无变化
  return x
def g_trans_non(x,y):#导数
  return 1
def trans_inv_non(y):#逆变换
  return y

def trans_a(x,a):#有下界
  return a-1+np.sqrt(x**2+1)
def g_trans_a(x,y,a):
  return x/(y+1-a)
def trans_inv_a(y,a):
  if y<=a : return 0.0
  return np.sqrt((y+1-a)**2-1)

def trans_b(x,b):#有上界
  return b+1-np.sqrt(x**2+1)
def g_trans_b(x,y,b):
  return x/(y-b-1)
def trans_inv_b(y,b):
  if y>=b : return 0.0
  return np.sqrt((-y+1+b)**2-1)

def trans_ab(x,a,b):
  return (b-a)/2*(np.sin(x)+1) + a
def g_trans_ab(x,y,a,b):
  return (b-a)/2*np.cos(x)
def trans_inv_ab(y,a,b):
  if y<=a : return -np.pi/2
  if y>=b : return np.pi/2
  if a==b : return a
  return np.arcsin((2*y-a-b)/(b-a))


def get_trans_f(bnds):
  fun_t = []
  for a,b in bnds:
    if a is None:
      if b is None:
        # (-inf,inf)
        fun_t.append(trans_non)
      else:
        # (-inf,b)
        fun_t.append(lambda x,b=b:trans_b(x,b))
    else :
      if b is None:
        # (a,inf)
        fun_t.append(lambda x,a=a:trans_a(x,a))
      else:
        fun_t.append(lambda x,a=a,b=b:trans_ab(x,a,b))
  return fun_t

def get_trans_f_inv(bnds):
  fun_inv_t = []
  for a,b in bnds:
    if a is None:
      if b is None:
        # (-inf,inf)
        fun_inv_t.append(lambda y:trans_inv_non(y))
      else:
        # (-inf,b)
        fun_inv_t.append(lambda y,b=b:trans_inv_b(y,b))
    else :
      if b is None:
        # (a,inf)
        fun_inv_t.append(lambda y,a=a:trans_inv_a(y,a))
      else:
        fun_inv_t.append(lambda y,a=a,b=b:trans_inv_ab(y,a,b))
  return fun_inv_t

def get_trans_f_g(bnds):
  fun_t = []
  for a,b in bnds:
    if a is None:
      if b is None:
        # (-inf,inf)
        fun_t.append(lambda x,y:g_trans_non(x,y))
      else:
        # (-inf,b)
        fun_t.append(lambda x,y,b=b:g_trans_b(x,y,b))
    else :
      if b is None:
        # (a,inf)
        fun_t.append(lambda x,y,a=a:g_trans_a(x,y,a))
      else:
        fun_t.append(lambda x,y,a=a,b=b:g_trans_ab(x,y,a,b))
  return fun_t

class Bounds(object):
  """
  bounds trans function

.. doctest::

  ```
  >>> bnds = Bounds([(None,None),(-2,None),(None,2),(-2,2)])
  >>> @bnds.trans_f_g
  ... def f(x):
  ...   return np.sum(x),np.ones_like(x)
  ... 
  >>> bnds.get_y(bnds.get_x([1.0,2.0,-2.0,0.0]))
  [1.0, 2.0, -2.0, 0.0]
  >>> f(bnds.get_x([1.0,2.0,-2.0,0.0]))[0]
  1.0
  
  ```

  """
  def __init__(self,bnds):
    self.fun_t = get_trans_f(bnds)
    self.g_fun_t = get_trans_f_g(bnds)
    self.fun_inv_t = get_trans_f_inv(bnds)
  def trans_f_g(self,f):
    def f_t(x):
      y = [y_i(i) for i,y_i in zip(x,self.fun_t)]
      ret,g_ret = f(y)
      return ret,np.array([gi*gf(xi,yi) for xi,yi,gi,gf in zip(x,y,g_ret,self.g_fun_t)])
    return f_t
  
  def trans_f(self,f):
    def g(x):
      y = [y_i(i) for i,y_i in zip(x,self.fun_t)]
      ret = f(y)
      return f(y)
    return g
  
  def get_y(self,x):
    y = [y_i(i) for i,y_i in zip(x,self.fun_t)]
    return y
  
  def get_x(self,y):
    x = [x_i(i) for i,x_i in zip(y,self.fun_inv_t)]
    return x


