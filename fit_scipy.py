#!/usr/bin/env python3
from model import Cache_Model,set_gpu_mem_growth,param_list,FCN
import tensorflow as tf
import time
import numpy as np
import json
from scipy.optimize import minimize,BFGS,basinhopping
from angle import cal_ang_file,EularAngle

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

def main():
  dtype = "float64"
  set_gpu_mem_growth()
  tf.keras.backend.set_floatx(dtype)
  with open("Resonances.json") as f:  
    config_list = json.load(f)
  fname = [["data/data4600_new.dat","data/Dst0_data4600_new.dat"],
       ["data/bg4600_new.dat","data/Dst0_bg4600_new.dat"],
       ["data/PHSP4600_new.dat","data/Dst0_PHSP4600_new.dat"]
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
  with tf.device('/device:GPU:0'):
    data = load_data("data")
    bg = load_data("bg")
    mcdata = load_data("PHSP")
    a = Cache_Model(config_list,0.768331,data,mcdata,bg=bg,batch=65000)
  
  try :
    with open("2need.json") as f:  
      param = json.load(f)
      a.set_params(param)
  except:
    pass
  s = json.dumps(a.get_params(),indent=2)
  print(s)
  #print(data,bg,mcdata)
  t = time.time()
  nll,g = a.cal_nll_gradient()#data_w,mcdata,weight=weights,batch=50000)
  print("nll:",nll,"Time:",time.time()-t)
  #exit()
  #print(a.get_params())
  #t = time.time()
  #with tf.device('/device:CPU:0'):
      #with tf.GradientTape() as tape:
        #nll = a.nll(data,bg,mcdata)
      #g = tape.gradient(nll,a.Amp.trainable_variables)
  #print("Time:",time.time()-t)
  #print(nll,g)
  
  fcn = FCN(a)# 1356*18
  #a_h = Cache_Model(config_list,0.768331,data,mcdata,bg=bg,batch=26000)
  #a_h.set_params(a.get_params())
  #f_h = FCN(a_h)
  args = {}
  args_name = []
  x0 = []
  bnds = []
  bounds_dict = {
      "Zc_4160_m0:0":(4.1,4.22),
      "Zc_4160_g0:0":(0,None)
  }
  for i in a.Amp.trainable_variables:
    args[i.name] = i.numpy()
    x0.append(i.numpy())
    args_name.append(i.name)
    if i.name in bounds_dict:
      bnds.append(bounds_dict[i.name])
    else:
      bnds.append((None,None))
    args["error_"+i.name] = 0.1
  now = time.time()
  callback = None#lambda x: print(list(zip(args_name,x)))
  with tf.device('/device:GPU:0'):
    #s = basinhopping(f.nll_grad,np.array(x0),niter=6,disp=True,minimizer_kwargs={"jac":True,"options":{"disp":True}})
    s = minimize(fcn.nll_grad,np.array(x0),method="L-BFGS-B",jac=True,bounds=bnds,callback=callback,options={"disp":1,"maxcor":80})
  print(s)
  print(time.time()-now)
  with open("final_params.json","w") as f:
    json.dump(a.get_params(),f,indent=2)
  print("\nend\n")

if __name__ == "__main__":
  main()
