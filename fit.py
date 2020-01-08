#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from tf_pwa.model import Cache_Model,set_gpu_mem_growth,param_list,FCN
from tf_pwa.angle import cal_ang_file,EularAngle
from tf_pwa.utils import load_config_file,flatten_np_data,pprint,error_print
from tf_pwa.fitfractions import cal_fitfractions
from iminuit import Minuit
import time
import json

'''def train_one_step(model, optimizer):
  nll,grads = model.cal_nll_gradient({})
  optimizer.apply_gradients(zip(grads, model.Amp.trainable_variables))
  return nll,grads'''

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

def fit(init_params="init_params.json",hesse=True,frac=True):
  dtype = "float64"
  w_bkg = 0.768331
  set_gpu_mem_growth()
  tf.keras.backend.set_floatx(dtype)
  config_list = load_config_file("Resonances")

  data, bg, mcdata = prepare_data(dtype=dtype)
  
  a = Cache_Model(config_list,w_bkg,data,mcdata,bg=bg,batch=65000)
  
  try :
    with open(init_params) as f:  
      param = json.load(f)
      print("using {}".format(init_params))
      if "value" in param:
        a.set_params(param["value"])
      else :
        a.set_params(param)
  except Exception as e:
    #print(e)
    print("using random parameters")
  
  pprint(a.get_params())

  '''
  if False: #check gradient
    data_w,weights = data,1.0#a.get_weight_data(data,bg)
    t = time.time()
    #print(a.Amp(data))
    #print(a.Amp(bg))
    #print(a.Amp(mcdata)
    #print(tf.reduce_sum(tf.math.log(a.Amp(data))))
    nll,g = a.cal_nll_gradient()#data_w,mcdata,weight=weights,batch=50000)
    print("Time:",time.time()-t)
    #print(nll)
    #he = np.array([[j.numpy() for j in i] for i in h])
    #print(he)
    #ihe = np.linalg.inv(he)
    #print(ihe)
    #exit()
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
  '''
   
  fcn = FCN(a)
  a.Amp.polar = False
  args = {}
  args_name = []
  x0 = []
  bounds_dict = {
      "Zc_4160_m0:0":(4.1,4.22),
      "Zc_4160_g0:0":(0,None)
  }
  for i in a.Amp.trainable_variables:
    args[i.name] = i.numpy()
    x0.append(i.numpy())
    args_name.append(i.name)
    if i.name in bounds_dict:
      args["limit_{}".format(i)] = bounds_dict[i]
    args["error_"+i.name] = 0.1
 

  m = Minuit(fcn,forced_parameters=args_name,errordef = 0.5,grad=fcn.grad,print_level=2,use_array_call=True,**args)

  now = time.time()
  with tf.device('/device:GPU:0'):
    m.migrad()#(ncall=10000))#,precision=5e-7))
  print("MIGRAD Time",time.time() - now)
  if hesse:
    now = time.time()
    with tf.device('/device:GPU:0'):
      m.hesse()
    print("HESSE Time",time.time() - now)
  '''now = time.time()
  with tf.device('/device:GPU:0'):
    m.minos(var="D2_2460_BLS_2_1r:0")
    m.minos(var="D1_2430_BLS_2_2i:0")
  print("MINOS Time",time.time() - now)'''
  print(m.values)
  print(m.errors)
  #print(m.get_param_states())

  err_mtrx=m.np_covariance()
  np.save("error_matrix.npy",err_mtrx)
  err=dict(m.errors)

  outdic={"value":a.get_params(),"error":err}
  with open("final_params.json","w") as f:
    json.dump(outdic,f,indent=2)

  print("##########")
  val = m.values
  i = 0
  for v in args_name:
    if len(v)>15:
      if i%2==0:
        tmp = val[v]
      else:
        if a.Amp.polar:
          print(v,"\t%.5f * exp(%.5fi)"%(tmp,val[v]))
        else:  
          rho = np.sqrt(val[v]**2+tmp**2)
          phi = np.arctan2(val[v],tmp)
          print(v,"\t%.5f * exp(%.5fi)"%(rho,phi))
    i+=1
  print("##########")

  if frac:
    mcdata_cached = a.Amp.cache_data(*mcdata,batch=65000)
    frac,grad = cal_fitfractions(a.Amp,mcdata_cached,kwargs={"cached":True})
    err_frac = {}
    for i in config_list:
      if hesse:
        err_frac[i] = np.sqrt(np.dot(np.dot(err_mtrx,grad[i]),grad[i]))
      else :
        err_frac[i] = None
    print("fitfractions")
    for i in config_list:
      print(i,":",error_print(frac[i],err_frac[i]))
  print("\nend\n")
  #try :
    #print(m.minos())
  #except RuntimeError as e:
    #print(e)
  #print(m.get_param_states())
  #with tf.device('/device:GPU:0'):
    #print(a.nll(data,bg,mcdata))#.collect_params())
  #print(a.Amp.trainable_variables)
  '''t = time.time()
  a_h = Cache_Model(a.Amp,0.768331,data,mcdata,bg=bg,batch=26000)
  a_h.set_params(a.get_params())
  nll,g,h = a_h.cal_nll_hessian()#data_w,mcdata,weight=weights,batch=50000)
  print("Time:",time.time()-t)
  print("NLL",nll)
  print("gradients",[i.numpy() for i in g])
  #print(h.numpy())
  inv_he = np.linalg.inv(h.numpy())
  print("hesse error:")
  pprint(dict(zip(args_name,np.sqrt(inv_he.diagonal()).tolist())))'''

def main():
  import argparse
  parser = argparse.ArgumentParser(description="simple fit scripts")
  parser.add_argument("--no-hesse", action="store_false", default=True,dest="hesse")
  parser.add_argument("--no-frac", action="store_false", default=True,dest="frac")
  parser.add_argument("--method", default="BFGS",dest="method")
  results = parser.parse_args()
  fit(hesse=results.hesse, frac=results.frac)
  
if __name__=="__main__":
  main()
