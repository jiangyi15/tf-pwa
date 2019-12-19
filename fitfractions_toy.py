#!/usr/bin/env python3

from amplitude import AllAmplitude
from model import Cache_Model,set_gpu_mem_growth,param_list,FCN
import tensorflow as tf
import time
import numpy as np
import json
from scipy.optimize import minimize,BFGS,basinhopping
from angle import cal_ang_file,EularAngle
import math

def error_print(x,err=None):
  if err is None:
    return ("{}").format(x)
  if err <= 0 or math.isnan(err):
    return ("{} ? {}").format(x,err)
  d = math.ceil(math.log10(err))
  b = 10**d
  b_err = err/b
  b_val = x/b
  if b_err < 0.355: #0.100 ~ 0.354
    dig = 2
  elif b_err < 0.950: #0.355 ~ 0.949
    dig = 1
  else: # 0.950 ~ 0.999
    dig = 0
  err = round(b_err,dig) * b
  x = round(b_val,dig)*b
  d_p = dig - d
  if d_p > 0:
    return ("{0:.%df} +/- {1:.%df}"%(d_p,d_p)).format(x,err)
  return ("{0:.0f} +/- {1:.0f}").format(x,err)

def error_print(x,err):
  return ("{:.7f} +/- {:.7f}").format(x,err)

def flatten_np_data(data):
  ret = {}
  for i in data:
    tmp = data[i]
    if isinstance(tmp,EularAngle):
      ret["alpha"+i[3:]] = tmp.alpha
      ret["beta"+i[3:]] = tmp.beta
      ret["gamma"+i[3:]] = tmp.gamma
    else :
      ret[i] = data[i]
  return ret

param_list = [
  "m_BC", "m_BD", "m_CD", 
  "beta_BC", "beta_B_BC", "alpha_BC", "alpha_B_BC",
  "beta_BD", "beta_B_BD", "alpha_BD", "alpha_B_BD", 
  "beta_CD", "beta_D_CD", "alpha_CD", "alpha_D_CD",
  "beta_BD_B","beta_BC_B","beta_BD_D","beta_CD_D",
  "alpha_BD_B","gamma_BD_B","alpha_BC_B","gamma_BC_B","alpha_BD_D","gamma_BD_D","alpha_CD_D","gamma_CD_D"
]

def pprint(x):
  s = json.dumps(x,indent=2)
  print(s)

def part_config(config,name=[]):
  if isinstance(name,str):
    print(name)
    return {name:config[name]}
  ret = {}
  for i in name:
    ret[i] = config[i]
  return ret

def main():
  dtype = "float64"
  set_gpu_mem_growth()
  tf.keras.backend.set_floatx(dtype)
  with open("Resonances.json") as f:  
    config_list = json.load(f)
  fname = [["../RooAllAmplitude/data/data4600_new.dat","data/Dst0_data4600_new.dat"],
       ["../RooAllAmplitude/data/bg4600_new.dat","data/Dst0_bg4600_new.dat"],
       ["../RooAllAmplitude/data/PHSP4600_new.dat","data/Dst0_PHSP4600_new.dat"]
  ]
  tname = ["data","bg","PHSP"]
  data_np = {}
  for i in range(3):
    data_np[tname[i]] = cal_ang_file(fname[i][0],dtype)
  m0_A = (data_np["data"]["m_A"]).mean()
  m0_B = (data_np["data"]["m_B"]).mean()
  m0_C = (data_np["data"]["m_C"]).mean()
  m0_D = (data_np["data"]["m_D"]).mean()  
  def load_data(name):
    dat = []
    tmp = flatten_np_data(data_np[name])
    for i in param_list:
      tmp_data = tf.Variable(tmp[i],name=i,dtype=dtype)
      dat.append(tmp_data)
    return dat
  with tf.device('/device:GPU:0'):
    data = load_data("data")
    bg = load_data("bg")
    mcdata = load_data("PHSP")
    a = Cache_Model(config_list,0.768331,data,mcdata,bg=bg,batch=26000)
  n_data = data[0].shape[0]
  sw = a.sw.numpy()
  a.Amp.m0_A = m0_A
  a.Amp.m0_B = m0_B
  a.Amp.m0_C = m0_C
  a.Amp.m0_D = m0_D
  try :
    with open("final_params.json") as f:  
      param = json.load(f)
      a.set_params(param)
  except:
    pass
  s = json.dumps(a.get_params(),indent=2)
  print("params:")
  print(s)
  #np.savetxt("filename.txt",a)
  try :
    Vij = np.load("error_matrix.npy")
  except :
    nll,g,h = a.cal_nll_hessian()#data_w,mcdata,weight=weights,batch=50000)
    inv_he = np.linalg.inv(h.numpy())
    Vij = inv_he
  mcdata_cached = a.Amp.cache_data(*mcdata,batch=65000)
  values = a.Amp.trainable_variables
  mean = [i.numpy() for i in values]
  cov = Vij
  num = 100
  toy_params = np.random.multivariate_normal(mean,cov,num)
  print("toy_params:",toy_params)
  def cal_fitfractions(params):
    a.set_params(dict(zip([i.name for i in values],params)))
    int_mc = sum_no_gradient(a.Amp,mcdata_cached,kwargs={"cached":True})
    res = [i for i in config_list]
    n_res = len(res)
    fitFrac = {}
    for i in range(n_res):
      for j in range(i,-1,-1):
        tmp_config_list = part_config(config_list,[res[i],res[j]])
        amp_tmp = AllAmplitude(tmp_config_list)
        amp_tmp.set_params(a.get_params())
        amp_tmp.m0_A = m0_A
        amp_tmp.m0_B = m0_B
        amp_tmp.m0_C = m0_C
        amp_tmp.m0_D = m0_D
        int_tmp = sum_no_gradient(amp_tmp,mcdata_cached,kwargs={"cached":True})
        name = res[i]+"x"+res[j]
        if i == j:
          fitFrac[res[i]] = (int_tmp/int_mc)
        else :
          fitFrac[name] = (int_tmp/int_mc) - fitFrac[res[i]] - fitFrac[res[j]]
    #print("check sum:",np.sum([fitFrac[i] for i in fitFrac]))
    print(fitFrac["D2_2460"])
    return fitFrac
  fitFrac = []
  for i in range(num):
    fitFrac_i = cal_fitfractions(toy_params[i])
    fitFrac.append(np.array([fitFrac_i[i] for i in fitFrac_i]))
  mean_fitFrac = cal_fitfractions(mean)
  names = [i for i in mean_fitFrac] 
  fitFrac = np.array(fitFrac)
  np.save("test_fitfrac",fitFrac)
  print("fitfractions:")
  for i in range(len(names)):
    err = np.sqrt(np.mean((fitFrac[:,i] - mean_fitFrac[names[i]])**2))
    s = error_print(mean_fitFrac[names[i]],err)
    print(names[i],":",s)

def sum_no_gradient(amp,data,weight=1.0,func=lambda x:x,args=(),kwargs={}):
  if isinstance(weight,float):
    weight = [weight] * len(data)
  nll = 0.0
  for i in range(len(data)):
    amp2s = amp(data[i],*args,**kwargs)
    l_a = func(amp2s)
    p_nll = tf.reduce_sum(tf.cast(weight[i],l_a.dtype) * l_a)
    nll += p_nll
  return nll.numpy()

def sum_gradient(amp,data,weight=1.0,func=lambda x:x,args=(),kwargs={}):
  n_variables = len(amp.trainable_variables)
  if isinstance(weight,float):
    weight = [weight] * len(data)
  nll = 0.0
  g = None
  for i in range(len(data)):
    with tf.GradientTape() as tape:
      amp2s = amp(data[i],*args,**kwargs)
      l_a = func(amp2s)
      p_nll = tf.reduce_sum(tf.cast(weight[i],l_a.dtype) * l_a)
    nll += p_nll
    a = tape.gradient(p_nll,amp.trainable_variables)
    if g is None:
      g = a
    else :
      for j in range(n_variables):
        g[j] += a[j]
  return nll.numpy(),np.array([i.numpy() for i in g])

if __name__ == "__main__":
  main()

