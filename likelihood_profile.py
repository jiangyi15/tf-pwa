#!/usr/bin/env python3
from model import Cache_Model,set_gpu_mem_growth,FCN
import tensorflow as tf
import time
import numpy as np
import json
from scipy.optimize import minimize,BFGS,basinhopping
from angle import cal_ang_file,EularAngle
from fit_scipy import error_print,flatten_np_data,pprint,param_list
import matplotlib.pyplot as plt
import math


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
  with tf.device('/device:CPU:0'):
    data = load_data("data")
    bg = load_data("bg")
    mcdata = load_data("PHSP")
    a = Cache_Model(config_list,0.768331,data,mcdata,bg=bg,batch=65000)
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
  
  def lklhd_prfl(fixed_var):
    args = {}
    args_name = []
    x0 = []
    bnds = []
    bounds_dict = {
        "D1_2420_BLS_2_1r:0": (None,None),#(6.1,6.1),
        "D1_2420_BLS_2_1i:0": (None,None),#(-1.9,-1.9),
        "D1_2420_BLS_2_2r:0": (None,None),#(8,8),
        "D1_2420_BLS_2_2i:0": (None,None),#(2.3,2.3),
        "D1_2420_d_BLS_2_1r:0": (None,None),#(2.9,2.9),
        "D1_2420_d_BLS_2_1i:0": (None,None),#(2.8,2.8),
        "D1_2430_BLS_2_1r:0": (None,None),#(2,2.),
        "D1_2430_BLS_2_1i:0": (None,None),#(2.9,2.9),
        "D1_2430_BLS_2_2r:0": (None,None),#(-2.6,-2.6),
        "D1_2430_BLS_2_2i:0": (None,None),#(4.2,4.2),
        "D1_2430_d_BLS_2_1r:0": (None,None),#(-0.2,-0.2),
        "D1_2430_d_BLS_2_1i:0": (None,None),#(-1,-1),
        "D2_2460_BLS_2_1r:0": (None,None),#(10,10),
        "D2_2460_BLS_2_1i:0": (None,None),#(0.8,0.8),
        "D2_2460_BLS_2_2r:0": (None,None),#(5,5),
        "D2_2460_BLS_2_2i:0": (None,None),#(4.3,4.3),
        "D2_2460_BLS_2_3r:0": (None,None),#(8,8),
        "D2_2460_BLS_2_3i:0": (None,None),#(1.1,1.1),
        "D2_2460_BLS_4_3r:0": (None,None),#(7,7),
        "D2_2460_BLS_4_3i:0": (None,None),#(6.5,6.5),
        "Zc_4025_BLS_2_1r:0": (None,None),#(-2,-2),
        "Zc_4025_BLS_2_1i:0": (None,None),#(0.9,0.9),
        "Zc_4025_d_BLS_2_1r:0": (None,None),#(20,20),
        "Zc_4025_d_BLS_2_1i:0": (None,None),#(-0.8,-0.8),
        "Zc_4025_d_BLS_2_2r:0": (None,None),#(32,32),
        "Zc_4025_d_BLS_2_2i:0": (None,None),#(-0.1,-0.1),
        "Zc_4160_BLS_2_1r:0": (None,None),#(1.5,1.5),
        "Zc_4160_BLS_2_1i:0": (None,None),#(7.1,7.1),
        "Zc_4160_d_BLS_2_1r:0": (None,None),#(2.0,2.0),
        "Zc_4160_d_BLS_2_1i:0": (None,None),#(0.24,0.24),
        "Zc_4160_d_BLS_2_2r:0": (None,None),#(0.1,0.1),
        "Zc_4160_d_BLS_2_2i:0": (None,None),#(2.8,2.8),
        "D1_2420_BLS_0_1r:0": 1.0,
        "D1_2420_BLS_0_1i:0": 0.0,
        "D1_2420_d_BLS_0_1r:0": 1.0,
        "D1_2420_d_BLS_0_1i:0": 0.0,
        "D1_2430_BLS_0_1r:0": 1.0,
        "D1_2430_BLS_0_1i:0": 0.0,
        "D1_2430_d_BLS_0_1r:0": 1.0,
        "D1_2430_d_BLS_0_1i:0": 0.0,
        "D2_2460_BLS_0_1r:0": 1.0,
        "D2_2460_BLS_0_1i:0": 0.0,
        "D2_2460_d_BLS_2_1r:0": 1.0,
        "D2_2460_d_BLS_2_1i:0": 0.0,
        "Zc_4025_BLS_0_1r:0": 1.0,
        "Zc_4025_BLS_0_1i:0": 0.0,
        "Zc_4025_d_BLS_0_1r:0": 1.0,
        "Zc_4025_d_BLS_0_1i:0": 0.0,
        "Zc_4160_BLS_0_1r:0": 1.0,
        "Zc_4160_BLS_0_1i:0": 0.0,
        "Zc_4160_d_BLS_0_1r:0": 1.0,
        "Zc_4160_d_BLS_0_1i:0": 0.0,
        "D1_2420r:0": (None,None),#(0.11,0.11),
        "D1_2420i:0": (None,None),#(5.4,5.4),
        "D1_2420pi:0": (None,None),#(2,2),
        "D1_2430r:0": (None,None),#(1.2,1.2),
        "D1_2430i:0": (None,None),#(9,9),
        "D1_2430pi:0": (None,None),#(4.5,4.5),
        "D2_2460pi:0": (None,None),#(2.4,2.4),
        "Zc_4025r:0": (fixed_var,fixed_var),#(0.51,0.51),
        "Zc_4025i:0": (None,None),#(-0.1,-0.1),
        "Zc_4160r:0": (None,None),#(2,2),
        "Zc_4160i:0": (None,None),#(4.1,4.1),
        "D2_2460r:0": 1.0,
        "D2_2460i:0": 0.0,
        
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
    
    '''for i in a.Amp.trainable_variables:
      if i.name == "Zc_4025r:0":
          print(i)
          i.assign(2.00)'''
    
    now = time.time()
    callback = None#lambda x: print(list(zip(args_name,x)))
    with tf.device('/device:GPU:0'):
      #s = basinhopping(f.nll_grad,np.array(x0),niter=6,disp=True,minimizer_kwargs={"jac":True,"options":{"disp":True}})
      # 优化器
      s = minimize(fcn.nll_grad,np.array(x0),method="L-BFGS-B",jac=True,bounds=bnds,callback=callback,options={"disp":1,"maxcor":100})
    print(s)
    
    '''a_h = Cache_Model(a.Amp,0.768331,data,mcdata,bg=bg,batch=26000)
    a_h.set_params(val)
    t = time.time()
    nll,g,h = a_h.cal_nll_hessian()#data_w,mcdata,weight=weights,batch=50000)
    print("Time:",time.time()-t)
    #print(nll)
    #print([i.numpy() for i in g])
    #print(h.numpy())'''
    return s#.fun

  x=np.arange(0,1,0.1)
  y=[]
  for v in x:
    y.append(lklhd_prfl(v).fun)
  print(x)
  print(y)
  plt.plot(x,y)
  plt.savefig("lklhd_prfl.png")
  plt.clf()
  print("\nend\n")

if __name__ == "__main__":
  t1=time.time()
  main()
  t2=time.time()
  print("*"*10,t2-t1)
