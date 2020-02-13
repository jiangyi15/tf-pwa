import functools
import tensorflow as tf
import numpy as np

def nll_grad(f, var, args=(), kwargs=None, options=None):
  kwargs = kwargs if kwargs is not None else {}
  options = options if options is not None else {}
  @functools.wraps(f)
  def f_w():
    with tf.GradientTape() as tape:
      ret = f(*args, **kwargs)
    g = tape.gradient(ret, var, unconnected_gradients="zero", **options)
    return ret, g #到底返回个啥？
  return f_w

def cal_fitfractions(amp, mcdata, res=None, args=(), kwargs=None):
  r"""
  defination:

  .. math::
    FF_{i} = \frac{\int |A_i|^2 d\Omega }{ \int |\sum_{i}A_i|^2 d\Omega }
    \approx \frac{\sum |A_i|^2 }{\sum|\sum_{i} A_{i}|^2}

  gradients:

  .. math::
    FF_{i,j} = \frac{\int 2Re(A_i A_j*) d\Omega }{ \int |\sum_{i}A_i|^2 d\Omega }
    = \frac{\int |A_i +A_j|^2  d\Omega }{ \int |\sum_{i}A_i|^2 d\Omega } - FF_{i} - FF_{j}

  hessians:

  .. math::
    \frac{\partial }{\partial \theta_i }\frac{f(\theta_i)}{g(\theta_i)} =
      \frac{\partial f(\theta_i)}{\partial \theta_i} \frac{1}{g(\theta_i)} -
      \frac{\partial g(\theta_i)}{\partial \theta_i} \frac{f(\theta_i)}{g^2(\theta_i)}

  """
  kwargs = kwargs if kwargs is not None else {}
  var = amp.trainable_variables
  # allvar = [i.name for i in var]
  cahced_res = amp.used_res
  if res is None:
    res = list(amp.res)
  n_res = len(res)
  fitFrac = {}
  err_fitFrac = {}
  g_fitFrac = [None]*n_res
  amp.set_used_res(res)
  int_mc, g_int_mc = sum_gradient(amp, mcdata, var=var, args=args, kwargs=kwargs)
  for i in range(n_res):
    for j in range(i, -1, -1):
      amp_tmp = amp
      if i == j:
        name = res[i]
        amp_tmp.set_used_res([res[i]])
      else:
        name = "{}x{}".format(res[i], res[j])
        amp_tmp.set_used_res([res[i], res[j]])
      int_tmp, g_int_tmp = sum_gradient(amp_tmp, mcdata, var=var, args=args, kwargs=kwargs)
      if i == j:
        fitFrac[name] = (int_tmp/int_mc)
        gij = g_int_tmp/int_mc - (int_tmp/int_mc) * g_int_mc/int_mc
        g_fitFrac[i] = gij
      else:
        fitFrac[name] = (int_tmp/int_mc) - fitFrac[res[i]] - fitFrac[res[j]]
        gij = g_int_tmp/int_mc - (int_tmp/int_mc) * g_int_mc/int_mc - g_fitFrac[i] - g_fitFrac[j]
      #print(name,gij.tolist())
      err_fitFrac[name] = gij
  amp.set_used_res(cahced_res)
  return fitFrac, err_fitFrac

def cal_fitfractions_no_grad(amp, mcdata, res=None, args=(), kwargs=None):
  r"""
  calculate fit fractions without gradients.
  """
  kwargs = kwargs if kwargs is not None else {}
  var = amp.trainable_variables
  # allvar = [i.name for i in var]
  cahced_res = amp.used_res
  if res is None:
    res = list(amp.res)
  n_res = len(res)
  fitFrac = {}
  amp.set_used_res(res)
  int_mc = sum_no_gradient(amp, mcdata, var=var, args=args, kwargs=kwargs)
  for i in range(n_res):
    for j in range(i, -1, -1):
      amp_tmp = amp
      if i == j:
        name = res[i]
        amp_tmp.set_used_res([res[i]])
      else:
        name = "{}x{}".format(res[i], res[j])
        amp_tmp.set_used_res([res[i], res[j]])
      int_tmp = sum_no_gradient(amp_tmp, mcdata, var=var, args=args, kwargs=kwargs)
      if i == j:
        fitFrac[name] = (int_tmp/int_mc)
      else:
        fitFrac[name] = (int_tmp/int_mc) - fitFrac[res[i]] - fitFrac[res[j]]
  amp.set_used_res(cahced_res)
  return fitFrac

def sum_gradient(amp, data, var, weight=1.0, func=lambda x: x, grad=True, args=(), kwargs=None):
  kwargs = kwargs if kwargs is not None else {}
  # n_variables = len(var)
  if isinstance(weight, float): #给data乘weight
    weight = [weight] * len(data)
  nll_list = []
  g_list = []

  def f(d, w):
    amp2s = amp(d, *args, **kwargs) #amp是振幅表达式
    l_a = func(amp2s) #ampPDF转换成每组数据的NLL表达式
    return tf.reduce_sum(tf.cast(w, l_a.dtype) * l_a)

  for d, w in zip(data, weight):
    if grad: #是否提供grad表达式
      p_nll, a = nll_grad(f, var, args=(d, w))() #调用上面定义的nll_grad，返回f(d,w)和f对var的导数
      g_list.append([i.numpy() for i in a])
    else:
      p_nll = f(d, w)
    nll_list.append(p_nll.numpy())
  nll = sum(nll_list)
  if grad:
    g = np.array(g_list).sum(0)
    return nll, g #nll值和各var的导数g值
  return nll

sum_no_gradient = functools.partial(sum_gradient, grad=False)
