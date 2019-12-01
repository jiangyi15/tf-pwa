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



class Model(object):
  def __init__(self,configs,w_bkg = 0,args=(),kwargs={}):
    self.w_bkg = w_bkg
    self.Amp = AllAmplitude(configs,*args,**kwargs)
  
  def get_weight_data(self,data,bg):
    n_data = data[0].shape[0]
    n_bg = data[0].shape[0]
    alpha = (n_data - self.w_bkg * n_bg)/(n_data + self.w_bkg**2 * n_bg)
    weight = [alpha]*n_data + [-self.w_bkg * alpha] * n_bg
    n_param  = len(data)
    data_warp = [tf.concat([data[i],bg[i]],0) for i in range(N)]
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
      nll = - nll_0 + sw * tf.math.log(int_mc/n_mc)
      g = [ -g0[i] + sw * g1[i]/int_mc for i in range(len(g0))]
      return nll,g
  
  def sum_gradient(self,data,weight=1.0,func = lambda x:x,args=(),kwargs={}):
    n_variables = len(self.Amp.trainable_variables)
    g = None
    nll = 0.0
    if isinstance(weight,float):
      weight = [weight] * len(data)
    for i in range(len(data)):
      #print(i,min(i*batch+batch,n_data))
      with tf.GradientTape() as tape:
        amp2s = self.Amp(data[i])
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

#class Model:
  #def __init__(self,res,w_bkg = 0):
    #self.Amp = AllAmplitude(res)
    #self.w_bkg = w_bkg
    
  #def nll(self,data,bg,mcdata):
    #ln_data = tf.reduce_sum(tf.math.log(self.Amp(data)))
    #ln_bg = tf.reduce_sum(tf.math.log(self.Amp(bg)))
    #int_mc = tf.math.log(tf.reduce_mean(self.Amp(mcdata)))
    #n_data = data[0].shape[0]
    #n_bg = bg[0].shape[0]
    #n_mc = mcdata[0].shape[0]
    #alpha = (n_data - self.w_bkg * n_bg)/(n_data + self.w_bkg**2 * n_bg)
    #return - alpha *(ln_data - self.w_bkg * ln_bg - (n_data - self.w_bkg*n_bg) * int_mc)
  
  #def nll_gradient(self,data,bg,mcdata,batch,cached=False,n_data=None,n_bg=None,n_mc=None,alpha=1.0):
    #if not cached:
      #n_data = data[0].shape[0]
      #n_bg = bg[0].shape[0]
      #n_mc = mcdata[0].shape[0]
      #alpha = (n_data - self.w_bkg * n_bg)/(n_data + self.w_bkg**2 * n_bg)
      #N = len(data)
      #data_warp = [tf.concat([data[i],bg[i]],0) for i in range(N)]
      #data_weight = tf.concat([tf.ones(shape=(n_data,),dtype=data[0].dtype),-tf.ones(shape=(n_bg,),dtype=data[0].dtype)*self.w_bkg],0)
      #sum_w = n_data - self.w_bkg * n_bg
      #nll,g = self.sum_gradient(data_warp,data_weight,batch,func=tf.math.log)
      #s,g2 = self.sum_gradient(mcdata,1/n_mc,batch)
      #for i in range(len(g)):
        #g[i] = -alpha* (g[i] - sum_w * g2[i]/s)
      #return -alpha*(nll - sum_w*tf.math.log(s)),g
    #else:
      #val = self.Amp.trainable_variables
      #with tf.GradientTape() as tape1:
        #ln_data = tf.reduce_sum(tf.math.log(self.Amp(data,cached)))
        #ln_bg = tf.reduce_sum(tf.math.log(self.Amp(bg,cached)))
        #nll_1 = - alpha *(ln_data - self.w_bkg * ln_bg)
      #g = tape1.gradient(nll_1,val)
      #int_mc = 0.0
      #g_1 = [0.0]*len(g)
      #for mcdata_i in mcdata:
        #with tf.GradientTape() as tape2:
          #amp2s = self.Amp(mcdata_i,cached)/n_mc
          #int_mc_i = tf.reduce_sum(amp2s)
        #g_2 = tape2.gradient(int_mc_i,val)
        #int_mc = int_mc + int_mc_i
        #for i in range(len(g_1)):
          #g_1[i] = g_1[i] +  g_2[i]
      #sum_w = alpha * (n_data - self.w_bkg * n_bg)
      #norm_mc = int_mc
      #for i in range(len(g)):
          #g[i] = g[i] + sum_w * g_1[i]/norm_mc
      #return nll_1 + sum_w *tf.math.log(norm_mc),g
  
  #def sum_gradient(self,data,weight=1.0,batch=1536,func = lambda x:x,cached=False):
    #data_i = []
    #N = len(data)
    #n_data = data[0].shape[0]
    #n_split = (n_data + batch -1) // batch
    #for i in range(n_split):
      #data_i.append([
        #data[j][i*batch:min(i*batch+batch,n_data)] for j in range(N)
      #])
    #data_wi = []
    #if isinstance(weight,float):
      #weight = tf.ones(n_data)*weight
    #for i in range(n_split):
      #data_wi.append(weight[i*batch:min(i*batch+batch,n_data)])
    #g = None
    #nll = 0.0
    #n_variables = len(self.Amp.trainable_variables)
    #for i in range(n_split):
      ##print(i,min(i*batch+batch,n_data))
      #with tf.GradientTape() as tape:
        #amp2s = self.Amp(data_i[i])
        #l_a = func(amp2s)
        #p_nll = tf.reduce_sum(tf.cast(data_wi[i],l_a.dtype) * l_a)
      #nll += p_nll
      #a = tape.gradient(p_nll,self.Amp.trainable_variables)
      #if g is None:
        #g = a
      #else :
        #for j in range(n_variables):
          #g[j] += a[j]
    #return nll,g
  
  #def get_params(self):
    #ret = {}
    #for i in self.Amp.variables:
      #tmp = i.numpy()
      #ret[i.name] = float(tmp)
    #return ret
  
  #def set_params(self,param):
    #for i in self.Amp.variables:
      #for j in param:
        #if j == i.name:
          #tmp = param[i.name]
          #i.assign(tmp)

param_list = [
  "m_BC", "m_BD", "m_CD", 
  "cos_BC", "cos_B_BC", "phi_BC", "phi_B_BC",
  "cos_BD", "cos_B_BD", "phi_BD", "phi_B_BD", 
  "cos_CD", "cos_D_CD", "phi_CD", "phi_D_CD",
  "cosbeta_B_BD","cosbeta_B_BC","cosbeta_D_BD","cosbeta_D_CD",
  "alpha_B_BD","gamma_B_BD","alpha_B_BC","gamma_B_BC","alpha_D_BD","gamma_D_BD","alpha_D_CD","gamma_D_CD"
]



def train_one_step(model, optimizer, data, bg,mc,batch=16384):
  nll,grads = model.nll_gradient(data,bg,mc,batch)
  print(grads)
  print(nll)
  optimizer.apply_gradients(zip(grads, model.Amp.trainable_variables))
  #with tf.GradientTape() as tape:
    #nll = model.nll(data,bg,mc)
  #g = tape.gradient(nll,model.Amp.trainable_variables)
  #print(nll,g)
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


