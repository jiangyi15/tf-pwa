import functools
import tensorflow as tf
import numpy as np

def nll_grad(f,var,**args):
  @functools.wraps(f)
  def f_w():
    with tf.GradientTape() as tape:
      ret = f()
    g = tape.gradient(ret,var,unconnected_gradients="zero")
    return ret,g
  return f_w

def cal_fitfractions(amp,mcdata,res=None,args=(),kwargs={}):
  r"""
  
  .. math::
    FFi = \frac{\int |A_i|^2 d\Omega }{ \int |\sum_{i}A_i|^2 d\Omega } 
    \approx \frac{\sum |A_i|^2 }{\sum|\sum_{i} A_{i}|^2} 

  .. math::
    FFij = \frac{\int 2Re(A_i A_j*) d\Omega }{ \int |\sum_{i}A_i|^2 d\Omega } 
    = \frac{\int |A_i +A_j|^2  d\Omega }{ \int |\sum_{i}A_i|^2 d\Omega } - FFi - FFj 
  
  .. math::
    \frac{\partial }{\partial \theta_i }\frac{f(\theta_i)}{g(\theta_i)} = 
      \frac{\partial f(\theta_i)}{\partial \theta_i} \frac{1}{g(\theta_i)} - 
      \frac{\partial g(\theta_i)}{\partial \theta_i} \frac{f(\theta_i)}{g^2(\theta_i)}
    
  """
  var = amp.trainable_variables
  allvar = [i.name for i in var]
  if res is None:
    res = [i for i in amp.res]
  n_res = len(res)
  fitFrac = {}
  err_fitFrac = {}
  g_fitFrac = [None]*n_res
  amp.set_used_res(res)
  int_mc,g_int_mc = sum_gradient(amp,mcdata,var=var,kwargs=kwargs)
  for i in range(n_res):
    for j in range(i,-1,-1):
      amp_tmp = amp
      if i == j :
        name = res[i]
        amp_tmp.set_used_res([res[i]])
      else :
        name = res[i]+"x"+res[j]
        amp_tmp.set_used_res([res[i],res[j]])
      int_tmp, g_int_tmp = sum_gradient(amp_tmp,mcdata,var=var,kwargs=kwargs)
      if i == j:
        fitFrac[name] = (int_tmp/int_mc)
        g_fitFrac[i]  = g_int_tmp/int_mc - (int_tmp/int_mc) * g_int_mc/int_mc
        gij = g_fitFrac[i]
      else :
        fitFrac[name] = (int_tmp/int_mc) - fitFrac[res[i]] - fitFrac[res[j]]
        gij  = g_int_tmp/int_mc - (int_tmp/int_mc) * g_int_mc/int_mc - g_fitFrac[i] - g_fitFrac[j]
      #print(name,gij.tolist())
      err_fitFrac[name] = gij
  return fitFrac,err_fitFrac

def sum_gradient(amp,data,weight=1.0,func=lambda x:x,var = [],args=(),kwargs={}):
  n_variables = len(var)
  if isinstance(weight,float):
    weight = [weight] * len(data)
  nll = 0.0
  g = None
  for i in range(len(data)):
    def f():
      amp2s = amp(data[i],*args,**kwargs)
      l_a = func(amp2s)
      return tf.reduce_sum(tf.cast(weight[i],l_a.dtype) * l_a)
    p_nll,a = nll_grad(f,var)()
    nll += p_nll
    if g is None:
      g = a
    else :
      for j in range(n_variables):
        g[j] += a[j]
  return nll.numpy(),np.array([i.numpy() for i in g])
