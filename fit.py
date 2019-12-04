#!/usr/bin/env python3

import json
from model import *


class fcn(object):
  def __init__(self,model,data,mcdata,batch=5000,bg=None):
    self.model = model
    if bg is None:
      self.data, self.weight = data, 1.0
    else:
      self.data, self.weight = model.get_weight_data(data,bg)
    self.mcdata = mcdata
    self.batch = batch
    
  def __call__(self,x):
    train_vars = self.model.Amp.trainable_variables
    n_var = len(train_vars)
    for i in range(n_var):
      train_vars[i].assign(x[i])
    nll,g = self.model.nll_gradient(self.data,self.mcdata,self.weight,batch=self.batch)
    return nll.numpy()
  
  def grad(self,x):
    now = time.time()
    train_vars = self.model.Amp.trainable_variables
    n_var = len(train_vars)
    for i in range(n_var):
      train_vars[i].assign(x[i])
    nll,g = self.model.nll_gradient(self.data,self.mcdata,self.weight,batch=self.batch)
    self.grads = [ i.numpy() for i in g]
    print("nll:", nll," time :",time.time() - now)
    return np.array(self.grads)

def train_one_step(model, optimizer):
  nll,grads = model.cal_nll_gradient({})
  optimizer.apply_gradients(zip(grads, model.Amp.trainable_variables))
  return nll,grads

def main():
  dtype = "float64"
  set_gpu_mem_growth()
  tf.keras.backend.set_floatx(dtype)
  with open("Resonances.json") as f:  
    config_list = json.load(f)
  def load_data(fname):
    dat = []
    with open(fname) as f:
      tmp = json.load(f)
      for i in param_list:
        tmp_data = tf.Variable(tmp[i],name=i,dtype=dtype)
        dat.append(tmp_data)
    return dat
  with tf.device('/device:GPU:0'):
    data = load_data("./data/data_ang_n4.json")
    bg = load_data("./data/bg_ang_n4.json")
    mcdata = load_data("./data/PHSP_ang_n4.json")
    a = Cache_Model(config_list,0.768331,data,mcdata,bg=bg,batch=65000)
  #print(a.Amp.coef)
  
  try :
    with open("need.json") as f:  
      param = json.load(f)
      a.set_params(param)
  except:
    pass
  s = json.dumps(a.get_params(),indent=2)
  print(s)
  #a.Amp(data)
  #exit()
  data_w,weights = data,1.0#a.get_weight_data(data,bg)
  t = time.time()
  nll,g = a.cal_nll_gradient()#data_w,mcdata,weight=weights,batch=50000)
  print("Time:",time.time()-t)
  print(nll)
  #he = np.array([[j.numpy() for j in i] for i in h])
  #print(he)
  #ihe = np.linalg.inv(he)
  #print(ihe)
  if False: #check gradient
    print(g)
    ptr = 0
    for i in a.Amp.trainable_variables:
      tmp = i.numpy()
      i.assign(tmp+1e-3)
      nll_0,g0 = a.nll_gradient(data_w,mcdata,weight=weights,batch=50000)
      i.assign(tmp-1e-3)
      nll_1,g1 = a.nll_gradient(data_w,mcdata,weight=weights,batch=50000)
      i.assign(tmp)
      print(i,(nll_0-nll_1).numpy()/2e-3)
      print(he[ptr])
      ptr+=1
      for j in range(len(g0)):
        print((g0[j]-g1[j]).numpy()/2e-3)
  
  import iminuit 
  f = FCN(a)
  args = {}
  args_name = []
  x0 = []
  for i in a.Amp.trainable_variables:
    args[i.name] = i.numpy()
    x0.append(i.numpy())
    args_name.append(i.name)
    args["error_"+i.name] = 0.1
  #args["limit_Zc_4160_m0:0"] = (4.1,4.22)
  m = iminuit.Minuit(f,forced_parameters=args_name,errordef = 0.5,grad=f.grad,print_level=2,use_array_call=True,**args)
  now = time.time()
  with tf.device('/device:GPU:0'):
    print(m.migrad(ncall=10000))#,precision=5e-7))
  print(time.time() - now)
  print(m.get_param_states())
  with open("final_params2.json","w") as f:
    json.dump(a.get_params(),f,indent=2)
  #try :
    #print(m.minos())
  #except RuntimeError as e:
    #print(e)
  #print(m.get_param_states())
  #with tf.device('/device:GPU:0'):
    #print(a.nll(data,bg,mcdata))#.collect_params())
  #print(a.Amp.trainable_variables)
  #t = time.time()
  #nll,g,h = a.cal_nll_hessian()#data_w,mcdata,weight=weights,batch=50000)
  #print("Time:",time.time()-t)
  #print(nll)
  #print(g)
  #print(h)
  
if __name__=="__main__":
  main()
