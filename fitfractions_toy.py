#!/usr/bin/env python3

from tf_pwa.amplitude import AllAmplitude
from tf_pwa.model import Cache_Model,param_list,FCN
import tensorflow as tf
import time
import numpy as np
import json
from scipy.optimize import minimize,BFGS,basinhopping
from tf_pwa.angle import cal_ang_file
from tf_pwa.utils import load_config_file,error_print,pprint,flatten_np_data
from tf_pwa.fitfractions import cal_fitfractions_no_grad as cal_frac
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
  #set_gpu_mem_growth()
  tf.keras.backend.set_floatx(dtype)
  config_list = load_config_file("Resonances")
  fname = [["./data/data4600_new.dat","data/Dst0_data4600_new.dat"],
       ["./data/bg4600_new.dat","data/Dst0_bg4600_new.dat"],
       ["./data/PHSP4600_new.dat","data/Dst0_PHSP4600_new.dat"]
  ]
  tname = ["data","bg","PHSP"]
  data_np = {}
  for i in range(3):
    data_np[tname[i]] = cal_ang_file(fname[i][0],dtype)
    
  def load_data(name):
    dat = []
    tmp = flatten_np_data(data_np[name])
    for i in param_list:
      tmp_data = tf.Variable(tmp[i],name=i,dtype=dtype)
      dat.append(tmp_data)
    return dat
  #with tf.device('/device:GPU:0'):
  data = load_data("data")
  bg = load_data("bg")
  mcdata = load_data("PHSP")
  a = Cache_Model(config_list,0.768331,data,mcdata,bg=bg,batch=26000)
  n_data = data[0].shape[0]
  
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
  num = 1000
  toy_params = np.random.multivariate_normal(mean,cov,num)
  print("toy_params:",toy_params)
  def cal_fitfractions(params):
    res = [i for i in config_list]
    a.set_params(dict(zip([i.name for i in values],params)))
    fitFrac = cal_frac(a.Amp,mcdata_cached,kwargs={"cached":True})
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


if __name__ == "__main__":
  main()

