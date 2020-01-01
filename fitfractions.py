#!/usr/bin/env python3

from tf_pwa.amplitude import AllAmplitude
from tf_pwa.model import Cache_Model,set_gpu_mem_growth,param_list,FCN
import tensorflow as tf
import time
import numpy as np
import json
from scipy.optimize import minimize,BFGS,basinhopping
from tf_pwa.angle import cal_ang_file
from tf_pwa.utils import load_config_file,error_print,flatten_np_data
import math


def error_print(x,err):
  return ("{:.7f} +/- {:.7f}").format(x,err)


param_list = [
  "m_A","m_B","m_C","m_D","m_BC", "m_BD", "m_CD", 
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

def cal_fitfractions(amp,mcdata,args=(),kwargs={}):
  allvar = [i.name for i in amp.trainable_variables]
  res = [i for i in amp.res]
  n_res = len(res)
  fitFrac = {}
  err_fitFrac = {}
  g_fitFrac = [None]*n_res
  amp.set_used_res(res)
  int_mc,g_int_mc = sum_gradient(amp,mcdata,kwargs=kwargs)
  for i in range(n_res):
    for j in range(i,-1,-1):
      amp_tmp = amp
      if i==j :
        name = res[i]
        amp_tmp.set_used_res([res[i]])
      else :
        name = res[i]+"x"+res[j]
        amp_tmp.set_used_res([res[i],res[j]])
      int_tmp, g_int_tmp = sum_gradient(amp_tmp,mcdata,kwargs=kwargs)
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

def err_trans(grad,Vij):
  return np.dot(grad.T,np.dot(Vij,grad))



def main():
  dtype = "float64"
  set_gpu_mem_growth()
  tf.keras.backend.set_floatx(dtype)
  config_list = load_config_file("Resonances")
  fname = [["./data/data4600_new.dat","data/Dst0_data4600_new.dat"],
       ["./data/bg4600_new.dat","data/Dst0_bg4600_new.dat"],
       ["./data/PHSP4600_new.dat","data/Dst0_PHSP4600_new.dat"],
       #["./data/PHSP_NEFF4600.dat",""]
  ]
  tname = ["data","bg","PHSP","MC"]
  data_np = {}
  for i in range(len(fname)):
    data_np[tname[i]] = cal_ang_file(fname[i][0],dtype)
  
  def load_data(name):
    dat = []
    tmp = flatten_np_data(data_np[name])
    for i in param_list:
      tmp_data = tf.Variable(tmp[i],name=i,dtype=dtype)
      dat.append(tmp_data)
    return dat
  
  data = load_data("data")
  bg = load_data("bg")
  mcdata = load_data("PHSP")
  #flat_mc_data = load_data("MC")
  a = Cache_Model(config_list,0.768331,data,mcdata,bg=bg,batch=26000)
  n_data = data[0].shape[0]
  sw = a.sw.numpy()
  
  with open("final_params.json") as f:  
    param = json.load(f)
    a.set_params(param)
  
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
  #flat_mc_data_cached = a.Amp.cache_data(*flat_mc_data,batch=65000)
  allvar = [i.name for i in a.Amp.trainable_variables]
  print(dict(zip(allvar,np.sqrt(np.diag(Vij).tolist()))))
  
  fitFrac,g_fitFrac = cal_fitfractions(a.Amp,mcdata_cached,kwargs={"cached":True})
  #fitFrac_f,g_fitFrac_f = cal_fitfractions(a.Amp,flat_mc_data_cached,kwargs={"cached":True})
  print("check sum:",np.sum([fitFrac[i] for i in fitFrac]))
  print("fitfractions:")
  for i in fitFrac:
    s = error_print(fitFrac[i], np.sqrt(err_trans(g_fitFrac[i],Vij)))
    print(i,":",s)
  
  #print("\nphysics fitfractions:")
  #for i in fitFrac_f:
    #s = error_print(fitFrac_f[i], np.sqrt(err_trans(g_fitFrac_f[i],Vij)))
    #print(i,":",s)

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
    a = tape.gradient(p_nll,amp.trainable_variables,unconnected_gradients="zero")
    if g is None:
      g = a
    else :
      for j in range(n_variables):
        g[j] += a[j]
  return nll.numpy(),np.array([i.numpy() for i in g])

if __name__ == "__main__":
  main()


