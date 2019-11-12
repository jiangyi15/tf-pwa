import tensorflow as tf
from amplitude import AllAmplitude
import time 
import functools

class Model:
  def __init__(self,res,w_bkg = 0):
    self.Amp = AllAmplitude(res)
    self.w_bkg = w_bkg
    
  def nll(self,data,bg,mcdata):
    ln_data = tf.reduce_sum(tf.math.log(self.Amp(data)))
    ln_bg = tf.reduce_sum(tf.math.log(self.Amp(bg)))
    int_mc = tf.math.log(tf.reduce_mean(self.Amp(mcdata)))
    n_data = data[0].shape[0]
    n_bg = bg[0].shape[0]
    n_mc = mcdata[0].shape[0]
    return -(ln_data - self.w_bkg * ln_bg - (n_data - self.w_bkg*n_bg) * int_mc)
  
  def nll_gradient(self,data,bg,mcdata,batch):
    n_data = data[0].shape[0]
    n_bg = bg[0].shape[0]
    n_mc = mcdata[0].shape[0]
    N = len(data)
    data_warp = [tf.concat([data[i],bg[i]],0) for i in range(N)]
    data_weight = tf.concat([tf.ones(shape=(n_data,),dtype="float32"),-tf.ones(shape=(n_bg,),dtype="float32")*self.w_bkg],0)
    sum_w = n_data - self.w_bkg * n_bg
    nll,g = self.sum_gradient(data_warp,data_weight,batch,func=tf.math.log)
    s,g2 = self.sum_gradient(mcdata,1/n_mc,batch)
    for i in range(len(g)):
      g[i] = -g[i] + sum_w * g2[i]/s
    return -nll+sum_w*tf.math.log(s),g
  
  def sum_gradient(self,data,weight=1.0,batch=1536,func = lambda x:x):
    data_i = []
    N = len(data)
    n_data = data[0].shape[0]
    n_split = (n_data + batch -1) // batch
    for i in range(n_split):
      data_i.append([
        data[j][i*batch:min(i*batch+batch,n_data)] for j in range(N)
      ])
    data_wi = []
    if isinstance(weight,float):
      weight = tf.ones(n_data)*weight
    for i in range(n_split):
      data_wi.append(weight[i*batch:min(i*batch+batch,n_data)])
    g = None
    nll = 0.0
    n_variables = len(self.Amp.trainable_variables)
    for i in range(n_split):
      #print(i,min(i*batch+batch,n_data))
      with tf.GradientTape() as tape:
        amp2s = self.Amp(data_i[i])
        l_a = func(amp2s)
        p_nll = tf.reduce_sum(data_wi[i] * l_a)
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
    for i in self.Amp.trainable_variables:
      tmp = i.numpy()
      ret[i.name] = float(tmp)
    return ret
  
  def set_params(self,param):
    for i in self.Amp.trainable_variables:
      tmp = param[i.name]
      i.assign(tmp)

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

class fcn(object):
  """
  provide FCN function and gradient for minuit
  """
  def __init__(self,model,data,bg,mc,batch=16384):
    self.model = model
    self.data = data
    self.bg = bg
    self.mc = mc
    self.batch = batch
    self.grads = []
    self.x = None
    self.nll = 0.0
    w_bkg = model.w_bkg
    n_data = data[0].shape[0]
    n_bg = bg[0].shape[0]
    self.alpha = (n_data - w_bkg * n_bg)/(n_data + w_bkg**2 * n_bg)
  def __call__(self,*x):
    now = time.time()
    if (not self.x is None) and self.x == x:
      return self.nll
    self.x = x
    train_vars = self.model.Amp.trainable_variables
    n_var = len(train_vars)
    for i in range(n_var):
      train_vars[i].assign(x[i])
    nll,g = self.model.nll_gradient(self.data,self.bg,self.mc,self.batch)
    self.grads = [ self.alpha * i.numpy() for i in g]
    print("nll:",self.alpha * nll," time :",time.time() - now)
    return self.alpha * nll
  
  @functools.lru_cache()
  def grad(self,*x):
    if (not self.x is None) and self.x == x:
      return self.grads
    self(*x)
    return self.grads
  
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

def main():
  import json
  set_gpu_mem_growth()
  with open("Resonances.json") as f:  
    config_list = json.load(f)
  a = Model(config_list,0.768331)
  #with open("test.json") as f:  
    #param = json.load(f)
    #a.set_params(param)
  s = json.dumps(a.get_params(),indent=2)
  print(s)
  def load_data(fname):
    dat = []
    with open(fname) as f:
      tmp = json.load(f)
      for i in param_list:
        tmp_data = tf.Variable(tmp[i],name=i)
        dat.append(tmp_data)
    return dat
  data = load_data("./data/data_ang_n4.json")
  bg = load_data("./data/bg_ang_n4.json")
  mcdata = load_data("./data/PHSP_ang_n4.json")
  #print(data,bg,mcdata)
  #a.Amp(data)
  #exit()
  #print(a.get_params())
  #t = time.time()
  #with tf.device('/device:GPU:0'):
    #print("NLL:",a.nll(data,bg,mcdata))#.collect_params())
  #print("Time:",time.time()-t)
  import iminuit 
  f = fcn(a,data,bg,mcdata,6780)# 1356*18
  args = {}
  args_name = []
  for i in a.Amp.trainable_variables:
    args[i.name] = i.numpy()
    args_name.append(i.name)
    args["error_"+i.name] = 0.1
  m = iminuit.Minuit(f,forced_parameters=args_name,errordef = 0.5,print_level=2,grad=f.grad,**args)
  now = time.time()
  with tf.device('/device:GPU:0'):
    m.migrad()
  print(time.time() - now)
  m.get_param_states()
  m.hesse()
  m.get_param_states()
  with tf.device('/device:GPU:0'):
    print(a.nll(data,bg,mcdata))#.collect_params())
  print(a.Amp.trainable_variables)
  
if __name__=="__main__":
  main()
