#!/usr/bin/env python3
from model import Cache_Model,set_gpu_mem_growth,FCN
import tensorflow as tf
import time
import numpy as np
import json
from scipy.optimize import minimize,BFGS,basinhopping
import iminuit
from angle import cal_ang_file,EularAngle
from fit import flatten_np_data,pprint,param_list
import matplotlib.pyplot as plt
from bounds import Bounds
import math


def main(param_name,x,method):
  dtype = "float64"
  w_bkg = 0.768331
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
    a = Cache_Model(config_list,w_bkg,data,mcdata,bg=bg,batch=65000)
  a.Amp.m0_A = m0_A
  a.Amp.m0_B = m0_B
  a.Amp.m0_C = m0_C
  a.Amp.m0_D = m0_D
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
  fcn = FCN(a)# 1356*18
  
  def LP_minuit(param_name,fixed_var):
    args = {}
    args_name = []
    x0 = []
    bounds_dict = {
        param_name: (fixed_var,fixed_var),

	"Zc_4160_m0:0":(4.1,4.22),
        "Zc_4160_g0:0":(0,10)
    }
    for i in a.Amp.trainable_variables:
      args[i.name] = i.numpy()
      x0.append(i.numpy())
      args_name.append(i.name)
      args["error_"+i.name] = 0.1
      if i.name not in bounds_dict:
        bounds_dict[i.name]=(0.,None)
    for i in bounds_dict:
      if i in args_name:
        args["limit_{}".format(i)] = bounds_dict[i]
    m = iminuit.Minuit(fcn,forced_parameters=args_name,errordef = 0.5,grad=fcn.grad,print_level=2,use_array_call=True,**args)
    now = time.time()
    with tf.device('/device:GPU:0'):
      print(m.migrad(ncall=10000))#,precision=5e-7))
    print(time.time() - now)
    print(m.get_param_states())
    return m

  def LP_sp(param_name,fixed_var):
    args = {}
    args_name = []
    x0 = []
    bnds = []
    bounds_dict = {
        param_name: (fixed_var,fixed_var),
        
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
        bnds.append((0.,None))
      args["error_"+i.name] = 0.1
    now = time.time()
    bd = Bounds(bnds)
    f_g = bd.trans_f_g(fcn.nll_grad)
    callback = lambda x: print(fcn.cached_nll)
    with tf.device('/device:GPU:0'):
      #s = basinhopping(f.nll_grad,np.array(x0),niter=6,disp=True,minimizer_kwargs={"jac":True,"options":{"disp":True}})
      # 优化器
      #s = minimize(fcn.nll_grad,np.array(x0),method="L-BFGS-B",jac=True,bounds=bnds,callback=callback,options={"disp":1,"maxcor":100})
      s = minimize(f_g,np.array(bd.get_x(x0)),method="BFGS",jac=True,callback=callback,options={"disp":1})
    return s

  #x=np.arange(0.51,0.52,0.01)
  y=[]
  if method=="scipy":
    for v in x:
      y.append(LP_sp(param_name,v).fun)
  elif method=="iminuit":
    for v in x:
      y.append(LP_minuit(param_name,v).get_fmin().fval)
  print("lklhdx",x)
  print("lklhdy",y)
  #plt.plot(x,y)
  #plt.savefig("lklhd_prfl.png")
  #plt.clf()
  print("\nend\n")
  return y

if __name__ == "__main__":
  param_name="D1_2420_BLS_2_1r:0" ###
  with open("final_params.json") as f:
    params = json.load(f)
  x_mean = params[param_name]
  #x_sigma = 
  x=np.arange(0.,10,0.5) ###
  method="scipy" ###
  t1=time.time()
  yf=main(param_name,x,method)
  t2=time.time()
  yb=main(param_name,x[::-1],method)
  t3=time.time()
  print("#"*10,t2-t1,"#"*10,t3-t2)
  plt.plot(x,yf,"*-",x,yb,"*-")
  plt.title(param_name)
  plt.legend(["forward","backward"])
  plt.savefig("lklhd_prfl")
  plt.clf()

