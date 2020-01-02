"""
basic negative log-likelihood model
"""

import tensorflow as tf
from .amplitude import AllAmplitude
import time 
import functools
import numpy as np

tf_version = int(tf.__version__.split(".")[0])
if tf_version < 2:
  tf.compat.v1.enable_eager_execution()

def array_split(data,batch=None):
  if batch is None:
    return [data]
  ret = []
  n_data = data[0].shape[0]
  n_split = (n_data + batch-1)//batch
  for i in range(n_split):
    tmp = []
    for data_i in data:
      tmp.append(data_i[i*batch:min(i*batch+batch,n_data)])
    ret.append(tmp)
  return ret


def time_print(f):
  @functools.wraps(f)
  def g(*args,**kwargs):
    now = time.time()
    ret = f(*args,**kwargs)
    print(f.__name__ ," cost time:",time.time()-now)
    return ret
  return g

class Model(object):
  """
  simple negative log likelihood model
  
  """
  def __init__(self,configs,w_bkg = 0,constrain={},args=(),kwargs={}):
    self.w_bkg = w_bkg
    if callable(configs):
      self.Amp = configs
    else :
      self.Amp = AllAmplitude(configs,*args,**kwargs)
    self.constrain = constrain
  
  def get_constrain_term(self):
    r"""
    constrain: Gauss(mean,sigma) 
      by add a term :math:`\frac{(\theta_i-\bar{\theta_i})^2}{2\sigma^2}`
    
    """
    t_var = self.Amp.trainable_variables
    t_var_name = [i.name for i in t_var]
    var_dict = dict(zip(t_var_name,t_var))
    nll = 0.0
    for i in self.constrain:
      if not i in var_dict:
        break
      pi = self.constrain[i]
      if isinstance(pi,tuple) and len(pi)==2:
        mean, sigma = pi
        var = var_dict[i]
        nll += (var - mean)**2/(sigma**2)/2
    return nll
  
  def get_constrain_grad(self):
    r"""
    constrain: Gauss(mean,sigma) 
      by add a term :math:`\frac{d}{d\theta_i}\frac{(\theta_i-\bar{\theta_i})^2}{2\sigma^2} = \frac{\theta_i-\bar{\theta_i}}{\sigma^2}`
    
    """
    t_var = self.Amp.trainable_variables
    t_var_name = [i.name for i in t_var]
    var_dict = dict(zip(t_var_name,t_var))
    g_dict = {}
    for i in self.constrain:
      if not i in var_dict:
        break
      pi = self.constrain[i]
      if isinstance(pi,tuple) and len(pi)==2:
        mean, sigma = pi
        var = var_dict[i]
        g_dict[i] = (var - mean)/(sigma**2)
    nll_g = []
    for i in t_var_name:
      if i in g_dict:
        nll_g.append(g_dict[i])
      else:
        nll_g.append(0.0)
    return nll_g
  
  def get_constrain_hessian(self):
    t_var = self.Amp.trainable_variables
    t_var_name = [i.name for i in t_var]
    var_dict = dict(zip(t_var_name,t_var))
    g_dict = {}
    for i in self.constrain:
      if not i in var_dict:
        break
      pi = self.constrain[i]
      if isinstance(pi,tuple) and len(pi)==2:
        mean, sigma = pi
        var = var_dict[i]
        g_dict[i] = 1/(sigma**2)
    nll_g = []
    for i in t_var_name:
      if i in g_dict:
        nll_g.append(g_dict[i])
      else:
        nll_g.append(0.0)
    return np.diag(nll_g)
  
  def get_weight_data(self,data,bg):
    if tf_version < 2:
      n_data = data[0].shape[0].value
      n_bg = bg[0].shape[0].value
    else :
      n_data = data[0].shape[0]
      n_bg = bg[0].shape[0]
    alpha = (n_data - self.w_bkg * n_bg)/(n_data + self.w_bkg**2 * n_bg)
    weight = [alpha]*n_data + [-self.w_bkg * alpha] * n_bg
    n_param  = len(data)
    data_warp = [tf.concat([data[i],bg[i]],0) for i in range(n_param)]
    return data_warp,weight

  def nll(self,data,mcdata,weight=1.0,batch=None,args=(),kwargs={}):
    r"""
    calculate negative log-likelihood 

    .. math::
      -\ln L = -\sum_{x_i \in data } w_i \ln f(x_i;\theta_i) +  (\sum w_i ) \ln \sum_{x_i \in mc } f(x_i;\theta_i) + cons

    """
    ln_data = tf.math.log(self.Amp(data))
    int_mc = tf.math.log(tf.reduce_mean(self.Amp(mcdata)))
    if isinstance(weight,float):
      weights = [weight]*len(data)
      sw = n_data * weight
    else :
      sw = tf.reduce_sum(weight)
      weights = [
        weight[i*batch:min(i*batch+batch,n_data)]
        for i in range(len(data))
      ]
    weights = tf.cast(weights,ln_data.dtype)
    nll_0 = - tf.reduce_sum(weights * ln_data)
    cons = self.get_constrain_term()
    return nll_0 + sw * int_mc + cons
  
  def nll_gradient(self,data,mcdata,weight=1.0,batch=None,args=(),kwargs={}):
    r"""
    calculate negative log-likelihood with gradient

    .. math::
      \frac{\partial }{\partial \theta_i }(-\ln L) = -\sum_{x_i \in data } w_i \frac{\partial }{\partial \theta_i } \ln f(x_i;\theta_i) +  
      \frac{\sum w_i }{\sum_{x_i \in mc }f(x_i;\theta_i)} \sum_{x_i \in mc } \frac{\partial }{\partial \theta_i } f(x_i;\theta_i) + cons
    
    """
    t_var = self.Amp.trainable_variables
    if batch is None:
      with tf.GradientTape() as tape:
        tape.watch(t_var)
        nll = self.nll(data,mcdata,weight)
      g = tape.gradient(nll,t_var)
      return nll,g
    else :
      n_data = data[0].shape[0]
      n_mc = mcdata[0].shape[0]
      data = array_split(data,batch)
      if isinstance(weight,float):
        weights = [weight]*len(data)
        sw = n_data * weight
      else :
        sw = tf.reduce_sum(weight)
        weights = [
          weight[i*batch:min(i*batch+batch,n_data)]
          for i in range(len(data))
        ]
      mcdata = array_split(mcdata,batch)
      nll_0,g0 = self.sum_gradient(data,weights,lambda x:tf.math.log(x))
      int_mc,g1 = self.sum_gradient(mcdata)
      sw = tf.cast(sw,nll_0.dtype)
      cons_grad = self.get_constrain_grad()
      cons = self.get_constrain_term()
      nll = - nll_0 + sw * tf.math.log(int_mc/tf.cast(n_mc,int_mc.dtype)) + cons
      g = [ cons_grad[i] - g0[i] + sw * g1[i]/int_mc for i in range(len(g0))]
      return nll,g
  
  def sum_hessian(self,data,weight=1.0,func = lambda x:x,args=(),kwargs={}):
    n_variables = len(self.Amp.trainable_variables)
    g = None
    h = None
    nll = 0.0
    if isinstance(weight,float):
      weight = [weight] * len(data)
    for i in range(len(data)):
      #print(i,min(i*batch+batch,n_data))
      with tf.GradientTape(persistent=True) as tape0:
        with tf.GradientTape() as tape:
          amp2s = self.Amp(data[i],*args,**kwargs)
          l_a = func(amp2s)
          p_nll = tf.reduce_sum(tf.cast(weight[i],l_a.dtype) * l_a)
        nll += p_nll
        a = tape.gradient(p_nll,self.Amp.trainable_variables)
      he = []
      for gi in a:
        he.append(tape0.gradient(gi,self.Amp.trainable_variables,unconnected_gradients="zero"))
      del tape0
      if g is None:
        g = a
        h = he
      else :
        for j in range(n_variables):
          g[j] += a[j]
          for k in range(n_variables):
            h[j][k] += he[j][k]
    return nll,g,h
  
  def sum_gradient(self,data,weight=1.0,func = lambda x:x,args=(),kwargs={}):
    n_variables = len(self.Amp.trainable_variables)
    g = None
    nll = 0.0
    if isinstance(weight,float):
      weight = [weight] * len(data)
    for i in range(len(data)):
      #print(i,min(i*batch+batch,n_data))
      with tf.GradientTape() as tape:
        tape.watch(self.Amp.trainable_variables)
        amp2s = self.Amp(data[i],*args,**kwargs)
        l_a = func(amp2s)
        p_nll = tf.reduce_sum(tf.cast(weight[i],l_a.dtype) * l_a)
      a = tape.gradient(p_nll,self.Amp.trainable_variables)
      nll += p_nll
      if g is None:
        g = a
      else :
        for j in range(n_variables):
          g[j] += a[j]
    return nll,g
  
  def get_params(self):
    ret = {}
    for i in self.Amp.variables:
      tmp = i.numpy()
      ret[i.name] = float(tmp)
    return ret
  
  def set_params(self,param):
    if isinstance(param,dict) and "value" in param:
      param = param["value"]
    var_list = self.Amp.variables
    var_name = [i.name for i in var_list]
    var = dict(zip(var_name,var_list))
    for i in param:
      if i in var_name:
        tmp = param[i]
        var[i].assign(tmp)

class Cache_Model(Model):
  def __init__(self,configs,w_bkg,data,mc,bg=None,batch=50000,constrain={},args=(),kwargs={}):
    super(Cache_Model,self).__init__(configs,w_bkg,constrain=constrain,*args,**kwargs)
    n_data = data[0].shape[0]
    self.n_mc = mc[0].shape[0]
    self.batch = batch
    if bg is None:
      weight = [1.0]*n_data
    else :
      data,weight = self.get_weight_data(data,bg)
      n_data = data[0].shape[0]
    self.data = self.Amp.cache_data(*data,batch=self.batch)
    self.mc = self.Amp.cache_data(*mc,batch=self.batch)
    self.sw = tf.reduce_sum(weight)
    self.weight = [
        weight[i*batch:min(i*batch+batch,n_data)]
        for i in range(len(self.data))
    ]
    self.init_params = self.get_params()
    self.t_var = self.Amp.trainable_variables
    self.t_var_name = [i.name for i in self.t_var]
    
  def cal_nll(self,params={}):
    if isinstance(params,dict):
      self.set_params(params)
    else:
      for i,j in zip(self.t_var,params):
        i.assign(j)
    nll_0 = self.sum_no_gradient(self.data,self.weight,lambda x:tf.math.log(x),kwargs={"cached":True})
    int_mc = self.sum_no_gradient(self.mc,kwargs={"cached":True})
    sw = tf.cast(self.sw,nll_0.dtype)
    nll = - nll_0 + sw * tf.math.log(int_mc/tf.cast(self.n_mc,int_mc.dtype))
    cons = self.get_constrain_term()
    return nll + cons
  
  def sum_no_gradient(self,data,weight=1.0,func = lambda x:x,args=(),kwargs={}):
    n_variables = len(self.Amp.trainable_variables)
    g = None
    nll = 0.0
    if isinstance(weight,float):
      weight = [weight] * len(data)
    for i in range(len(data)):
      amp2s = self.Amp(data[i],*args,**kwargs)
      l_a = func(amp2s)
      p_nll = tf.reduce_sum(tf.cast(weight[i],l_a.dtype) * l_a)
      nll += p_nll
    return nll
  
  def cal_nll_gradient(self,params={}):
    if isinstance(params,dict):
      self.set_params(params)
    else:
      for i,j in zip(self.Amp.trainable_variables,params):
        i.assign(j)
    nll_0,g0 = self.sum_gradient(self.data,self.weight,lambda x:tf.math.log(x),kwargs={"cached":True})
    int_mc,g1 = self.sum_gradient(self.mc,kwargs={"cached":True})
    sw = tf.cast(self.sw,nll_0.dtype)
    nll = - nll_0 + sw * tf.math.log(int_mc/tf.cast(self.n_mc,int_mc.dtype))
    cons = self.get_constrain_term()
    cons_grad = self.get_constrain_grad()
    g = [ cons_grad[i] -g0[i] + sw * g1[i]/int_mc for i in range(len(g0))]
    return nll + cons, g
  
  def cal_nll_hessian(self,params={}):
    if isinstance(params,dict):
      self.set_params(params)
    else:
      for i,j in zip(self.t_var,params):
        i.assign(j)
    nll_0,g0,h0 = self.sum_hessian(self.data,self.weight,lambda x:tf.math.log(x),kwargs={"cached":True})
    int_mc,g1,h1 = self.sum_hessian(self.mc,kwargs={"cached":True})
    cons = self.get_constrain_term()
    cons_grad = self.get_constrain_grad()
    cons_hessian = self.get_constrain_hessian()
    sw = tf.cast(self.sw,nll_0.dtype)
    nll = - nll_0 + sw * tf.math.log(int_mc/tf.cast(self.n_mc,int_mc.dtype))
    nll += cons
    g = [ cons_grad[i] - g0[i] + sw * g1[i]/int_mc for i in range(len(g0))]
    n = len(g0)
    h0 = np.array([[j.numpy() for j in i] for i in h0])
    h1 = np.array([[j.numpy() for j in i] for i in h1])
    g1 = np.array([i.numpy() for i in g1])/int_mc
    h = - h0 - sw *np.outer(g1,g1) + sw / int_mc * h1 + cons_hessian
    return nll, g, h
    
class FCN(object):
  def __init__(self,cache_model):
    self.model = cache_model
    self.ncall = 0
    self.n_grad = 0
    self.cached_nll = None
  #@time_print
  def __call__(self,x):
    nll = self.model.cal_nll(x)
    self.cached_nll = nll
    self.ncall += 1
    return nll
  
  @time_print
  def grad(self,x):
    nll,g = self.model.cal_nll_gradient(x)
    self.cached_nll = nll
    self.n_grad += 1
    return np.array([i.numpy() for i in g])
  
  #@time_print
  def nll_grad(self,x):
    nll,g = self.model.cal_nll_gradient(x)
    self.cached_nll = nll
    self.ncall += 1
    self.n_grad += 1
    return nll.numpy().astype("float64"), np.array([i.numpy() for i in g]).astype("float64")
  
  
  #@time_print
  def hessian(self,x):
    nll,g,h = self.model.cal_nll_hessian(x)
    self.cached_nll = nll
    return h
  
  def nll_grad_hessian(self,x):
    nll,g,h = self.model.cal_nll_hessian(x)
    self.cached_nll = nll
    return nll,g,h
  
    
param_list = [
  "m_A","m_B","m_C","m_D","m_BC", "m_BD", "m_CD", 
  "beta_BC", "beta_B_BC", "alpha_BC", "alpha_B_BC",
  "beta_BD", "beta_B_BD", "alpha_BD", "alpha_B_BD", 
  "beta_CD", "beta_D_CD", "alpha_CD", "alpha_D_CD",
  "beta_BD_B","beta_BC_B","beta_BD_D","beta_CD_D",
  "alpha_BD_B","gamma_BD_B","alpha_BC_B","gamma_BC_B","alpha_BD_D","gamma_BD_D","alpha_CD_D","gamma_CD_D"
]



def train_one_step(model, optimizer, data,mc,weight=1.0,batch=16384):
  nll,grads = model.nll_gradient(data,mc,weight,batch)
  print(grads)
  print(nll)
  optimizer.apply_gradients(zip(grads, model.Amp.trainable_variables))
  
  return nll

  
def set_gpu_mem_growth():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)


