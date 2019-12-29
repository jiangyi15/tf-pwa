#!/usr/bin/env python3
from model import Cache_Model,set_gpu_mem_growth,param_list,FCN
import tensorflow as tf
import time
import numpy as np
import json
from scipy.optimize import minimize,BFGS,basinhopping
from angle import cal_ang_file,EularAngle
from utils import load_config_file

import math
from bounds import Bounds

has_yaml = True
try:
  import yaml
except ImportError:
  has_yaml = False


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

def main():
  dtype = "float64"
  w_bkg = 0.768331
  set_gpu_mem_growth()
  tf.keras.backend.set_floatx(dtype)
  # open Resonances list as dict 
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
  a = Cache_Model(config_list,w_bkg,data,mcdata,bg=bg,batch=65000)#,constrain={"Zc_4160_g0:0":(0.1,0.1)})
  try :
    with open("init_params.json") as f:  
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
  
  fcn = FCN(a)
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
  import math
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
  
  bd = Bounds(bnds)
  f_g = bd.trans_f_g(fcn.nll_grad)
  points = []
  nlls = []
  def callback(x):
    points.append([float(i) for i in x])
    nlls.append(float(fcn.cached_nll))
    #print(fcn.cached_nll)
  #s = basinhopping(f.nll_grad,np.array(x0),niter=6,disp=True,minimizer_kwargs={"jac":True,"options":{"disp":True}})
  s = minimize(fcn.nll_grad,x0,method="L-BFGS-B",jac=True,bounds=bnds,callback=callback,options={"disp":1,"maxcor":100})
  #s = minimize(f_g,np.array(bd.get_x(x0)),method="BFGS",jac=True,callback=callback,options={"disp":1})
  xn = s.x
  #xn = bd.get_y(s.x)
  print(s)
  print(time.time()-now)
  val = dict(zip(args_name,xn))
  a.set_params(val)
  with open("final_params.json","w") as f:
    json.dump(a.get_params(),f,indent=2)
  with open("fit_curve.json","w") as f:
    json.dump({"points":points,"nlls":nlls},f,indent=2)
  a_h = Cache_Model(a.Amp,w_bkg,data,mcdata,bg=bg,batch=24000)
  a_h.set_params(val)
  t = time.time()
  nll,g,h = a_h.cal_nll_hessian()#data_w,mcdata,weight=weights,batch=50000)
  print("Time:",time.time()-t)
  #print(nll)
  #print([i.numpy() for i in g])
  #print(h.numpy())
  inv_he = np.linalg.pinv(h.numpy())
  diag_he = inv_he.diagonal()
  np.save("error_matrix.npy",inv_he)
  #print("edm:",np.dot(np.dot(inv_he,np.array(g)),np.array(g)))
  hesse_error = np.sqrt(diag_he).tolist()
  err = dict(zip(args_name,hesse_error))
  print("fit value:")
  for i in err:
    print("  ",i,":",error_print(val[i],err[i]))
  int_total = a.Amp(mcdata).numpy().sum()
  res_list = [i for i in config_list]
  fitFrac = {}
  for i in range(len(res_list)):
    name = res_list[i]
    a_sig = Cache_Model({name:config_list[name]},w_bkg,data,mcdata)
    a_sig.set_params(val)
    a_weight = a_sig.Amp(mcdata).numpy()
    fitFrac[name] = float(a_weight.sum()/int_total)
  print("FitFractions:")
  pprint(fitFrac)
  
  print("\nend\n")

if __name__ == "__main__":
  main()
