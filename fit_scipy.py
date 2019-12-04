from model import Cache_Model,set_gpu_mem_growth,param_list,FCN
from fit import fcn
import tensorflow as tf
import time
import numpy as np
import json
from scipy.optimize import minimize

if __name__=="__main__":
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
    with open("2need.json") as f:  
      param = json.load(f)
      a.set_params(param)
  except:
    pass
  s = json.dumps(a.get_params(),indent=2)
  #print(data,bg,mcdata)
  t = time.time()
  print(a.cal_nll())
  #exit()
  print("Time:",time.time()-t)
  #exit()
  #print(a.get_params())
  #t = time.time()
  #with tf.device('/device:CPU:0'):
      #with tf.GradientTape() as tape:
        #nll = a.nll(data,bg,mcdata)
      #g = tape.gradient(nll,a.Amp.trainable_variables)
  #print("Time:",time.time()-t)
  #print(nll,g)
  f = FCN(a)# 1356*18
  args = {}
  args_name = []
  x0 = []
  for i in a.Amp.trainable_variables:
    args[i.name] = i.numpy()
    x0.append(i.numpy())
    args_name.append(i.name)
    args["error_"+i.name] = 0.1
  now = time.time()
  callback = lambda x: print(list(zip(args_name,x)))
  with tf.device('/device:GPU:0'):
    s = minimize(f,np.array(x0),method="BFGS",jac=f.grad,callback=callback,options={"disp":True})
  print(s)
  print(time.time()-now)
  with open("final_params.json","w") as f:
    json.dump(a.get_params(),f,indent=2)
