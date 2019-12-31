#!/usr/bin/env python3
from tf_pwa.model import Cache_Model,set_gpu_mem_growth,param_list,FCN
import tensorflow as tf
import time
import numpy as np
import json
from scipy.optimize import minimize,BFGS,basinhopping
from tf_pwa.angle import cal_ang_file,EularAngle
from tf_pwa.utils import load_config_file,flatten_np_data,pprint,error_print

import math
from tf_pwa.bounds import Bounds

def cal_hesse_error(Amp,val,w_bkg,data,mcdata,bg,args_name,batch):
  a_h = Cache_Model(Amp,w_bkg,data,mcdata,bg=bg,batch=24000)
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
  return err

def prepare_data(dtype="float64"):
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
  return data, bg, mcdata

def cal_fitfractions(a,config_list,val,w_bkg,data,mcdata):
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

def fit(method="BFGS",hesse=True,frac=True):
  dtype = "float64"
  w_bkg = 0.768331
  set_gpu_mem_growth()
  tf.keras.backend.set_floatx(dtype)
  # open Resonances list as dict 
  config_list = load_config_file("Resonances")
  
  data, bg, mcdata = prepare_data(dtype=dtype)
  
  a = Cache_Model(config_list,w_bkg,data,mcdata,bg=bg,batch=65000)#,constrain={"Zc_4160_g0:0":(0.1,0.1)})
  try :
    with open("init_params.json") as f:  
      param = json.load(f)
      a.set_params(param["value"])
  except:
    pass
  
  pprint(a.get_params())
  #print(data,bg,mcdata)
  #t = time.time()
  #nll,g = a.cal_nll_gradient()#data_w,mcdata,weight=weights,batch=50000)
  #print("nll:",nll,"Time:",time.time()-t)
  
  fcn = FCN(a)
  
  # fit configure
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
  
  
  points = []
  nlls = []
  now = time.time()
  #s = basinhopping(f.nll_grad,np.array(x0),niter=6,disp=True,minimizer_kwargs={"jac":True,"options":{"disp":True}})
  if method == "BFGS":
    def callback(x):
      points.append([float(i) for i in bd.get_y(x)])
      nlls.append(float(fcn.cached_nll))
      print(fcn.cached_nll)
    bd = Bounds(bnds)
    f_g = bd.trans_f_g(fcn.nll_grad)
    s = minimize(f_g,np.array(bd.get_x(x0)),method="BFGS",jac=True,callback=callback,options={"disp":1})
    xn = bd.get_y(s.x)
  elif method == "L-BFGS-B":
    def callback(x):
      points.append([float(i) for i in x])
      nlls.append(float(fcn.cached_nll))
    s = minimize(fcn.nll_grad,x0,method="L-BFGS-B",jac=True,bounds=bnds,callback=callback,options={"disp":1,"maxcor":100})
    xn = s.x
  else :
    raise "unknow method"
  print("fit state:")
  print(s)
  print("Time for fitting:",time.time()-now)
  
  val = dict(zip(args_name,xn))
  a.set_params(val)
  
  with open("fit_curve.json","w") as f:
    json.dump({"points":points,"nlls":nlls},f,indent=2)

  err=None
  if hesse:
    err = cal_hesse_error(a.Amp,val,w_bkg,data,mcdata,bg,args_name,batch=24000)
  outdic={"value":a.get_params(),"error":err}
  with open("final_params.json","w") as f:
    json.dump(outdic,f,indent=2)
    
  print("fit value:")
  for i in val:
    if hesse:
      print("  ",i,":",error_print(val[i],err[i]))
    else:
      print("  ",i,":",val[i])
  if frac:
    cal_fitfractions(a,config_list,val,w_bkg,data,mcdata)
  

  
  print("\nend\n")

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="simple fit scripts")
  parser.add_argument("--no-hesse", action="store_false", default=True,dest="hesse")
  parser.add_argument("--no-frac", action="store_false", default=True,dest="frac")
  results = parser.parse_args()
  fit(method="BFGS",hesse=results.hesse,frac=results.frac)
