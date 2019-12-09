import tensorflow as tf
from amplitude import AllAmplitude
import time 
import functools
import numpy as np

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
  def __init__(self,configs,w_bkg = 0,args=(),kwargs={}):
    self.w_bkg = w_bkg
    if callable(configs):
      self.Amp = configs
    else :
      self.Amp = AllAmplitude(configs,*args,**kwargs)
  
  def get_weight_data(self,data,bg):
    n_data = data[0].shape[0]
    n_bg = bg[0].shape[0]
    alpha = (n_data - self.w_bkg * n_bg)/(n_data + self.w_bkg**2 * n_bg)
    weight = [alpha]*n_data + [-self.w_bkg * alpha] * n_bg
    n_param  = len(data)
    data_warp = [tf.concat([data[i],bg[i]],0) for i in range(n_param)]
    return data_warp,weight

  def nll(self,data,mcdata,weight=1.0,batch=None,args=(),kwargs={}):
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
    return nll_0 + sw * int_mc 
  
  def nll_gradient(self,data,mcdata,weight=1.0,batch=None,args=(),kwargs={}):
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
      nll = - nll_0 + sw * tf.math.log(int_mc/n_mc)
      g = [ -g0[i] + sw * g1[i]/int_mc for i in range(len(g0))]
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
        amp2s = self.Amp(data[i],*args,**kwargs)
        l_a = func(amp2s)
        p_nll = tf.reduce_sum(tf.cast(weight[i],l_a.dtype) * l_a)
      nll += p_nll
      a = tape.gradient(p_nll,self.Amp.trainable_variables)
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
    for i in self.Amp.variables:
      for j in param:
        if j == i.name:
          tmp = param[i.name]
          i.assign(tmp)

class Cache_Model(Model):
  def __init__(self,configs,w_bkg,data,mc,bg=None,batch=50000,args=(),kwargs={}):
    super(Cache_Model,self).__init__(configs,w_bkg,*args,**kwargs)
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
    nll = - nll_0 + sw * tf.math.log(int_mc/self.n_mc)
    return nll
  
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
      for i,j in zip(self.t_var,params):
        i.assign(j)
    nll_0,g0 = self.sum_gradient(self.data,self.weight,lambda x:tf.math.log(x),kwargs={"cached":True})
    int_mc,g1 = self.sum_gradient(self.mc,kwargs={"cached":True})
    sw = tf.cast(self.sw,nll_0.dtype)
    nll = - nll_0 + sw * tf.math.log(int_mc/self.n_mc)
    g = [ -g0[i] + sw * g1[i]/int_mc for i in range(len(g0))]
    return nll, g
  
  def cal_nll_hessian(self,params={}):
    if isinstance(params,dict):
      self.set_params(params)
    else:
      for i,j in zip(self.t_var,params):
        i.assign(j)
    nll_0,g0,h0 = self.sum_hessian(self.data,self.weight,lambda x:tf.math.log(x),kwargs={"cached":True})
    int_mc,g1,h1 = self.sum_hessian(self.mc,kwargs={"cached":True})
    
    sw = tf.cast(self.sw,nll_0.dtype)
    nll = - nll_0 + sw * tf.math.log(int_mc/self.n_mc)
    g = [ -g0[i] + sw * g1[i]/int_mc for i in range(len(g0))]
    n = len(g0)
    h0 = np.array([[j.numpy() for j in i] for i in h0])
    h1 = np.array([[j.numpy() for j in i] for i in h1])
    g1 = np.array([i.numpy() for i in g1])/int_mc
    h = - h0 - sw *np.outer(g1,g1) + sw / int_mc * h1
    return nll, g, h
    
class FCN(object):
  def __init__(self,cache_model):
    self.model = cache_model
  #@time_print
  def __call__(self,x):
    nll = self.model.cal_nll(x)
    return nll
  @time_print
  def grad(self,x):
    nll,g = self.model.cal_nll_gradient(x)
    return np.array([i.numpy() for i in g])
  
  @time_print
  def hessian(self,x):
    nll,g,h = self.model.cal_nll_hessian(x)
    return h

    
param_list = [
  "m_BC", "m_BD", "m_CD", 
  "cos_BC", "cos_B_BC", "phi_BC", "phi_B_BC",
  "cos_BD", "cos_B_BD", "phi_BD", "phi_B_BD", 
  "cos_CD", "cos_D_CD", "phi_CD", "phi_D_CD",
  "cosbeta_B_BD","cosbeta_B_BC","cosbeta_D_BD","cosbeta_D_CD",
  "alpha_B_BD","gamma_B_BD","alpha_B_BC","gamma_B_BC","alpha_D_BD","gamma_D_BD","alpha_D_CD","gamma_D_CD"
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


