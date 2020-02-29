#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from tf_pwa.model import Cache_Model,param_list,FCN
from tf_pwa.angle import cal_ang_file,EularAngle
from tf_pwa.utils import load_config_file,flatten_np_data,pprint,error_print,std_polar
from tf_pwa.fitfractions import cal_fitfractions
from tf_pwa.applications import fit_fractions, fit_minuit
#from iminuit import Minuit
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

def fit(init_params="init_params_rp.json",hesse=True,minos=False,frac=True):
  POLAR = True # fit in polar coordinates. should be consistent with init_params.json if any
  
  dtype = "float64"
  w_bkg = 0.768331
  #set_gpu_mem_growth()
  #tf.keras.backend.set_floatx(dtype)
  config_list = load_config_file("Resonances")

  data, bg, mcdata = prepare_data(dtype=dtype)
  
  a = Cache_Model(config_list,w_bkg,data,mcdata,bg=bg,batch=65000)
  if POLAR:
    print("Fitting parameters are defined in POLAR coordinates")
  else:
    print("Fitting parameters are defined in XY coordinates")
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
  a.Amp.trans_params(polar=POLAR)

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

  pprint(a.get_params())
  bounds_dict = {
    "Zc_4160_m0:0": (4.1, 4.22),
    "Zc_4160_g0:0": (0, None)
  }

  m = fit_minuit(a,bounds_dict=bounds_dict,hesse=True,minos=False)

  print("########## fit results")
  print(m.values)
  print(m.errors)
  #print(m.get_param_states())

  err_mtrx=m.np_covariance()
  np.save("error_matrix.npy",err_mtrx)
  err=dict(m.errors)

  params = a.get_params()
  '''print("\n########## fitting params in polar expression")
  i = 0
  for v in params:
    if len(v)>15:
      if i%2==0:
        tmp_name = v
        tmp = params[v]
      else:
        if POLAR:
          rho = tmp
          phi = params[v]
          rho,phi = std_polar(rho,phi)
        else:  
          rho = np.sqrt(params[v]**2+tmp**2)
          phi = np.arctan2(params[v],tmp)
        params[tmp_name] = rho
        params[v] = phi
        print(v[:-3],"\t%.5f * exp(%.5fi)"%(rho,phi))
      i+=1
  for v in config_list:
    rho = params[v.rstrip('pm')+'r:0']
    phi = params[v+'i:0']
    rho,phi = std_polar(rho,phi)
    params[v.rstrip('pm')+'r:0'] = rho
    params[v+'i:0'] = phi
    print(v,"\t\t%.5f * exp(%.5fi)"%(rho,phi))
  a.set_params(params)'''

  outdic={"value":params,"error":err}
  with open("final_params.json","w") as f:                                      
    json.dump(outdic,f,indent=2)


  if frac:
    frac, err_frac = fit_fractions(a,mcdata,config_list,inv_he,hesse)
    print("########## fit fractions")
    for i in config_list:
      print(i,":",error_print(frac[i],err_frac[i]))
  print("\nEND\n")
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
  #parser.add_argument("--yes-minos", action="store_true", default=False,dest="minos")
  parser.add_argument("--no-frac", action="store_false", default=True,dest="frac")
  results = parser.parse_args()
  fit(hesse=results.hesse, minos=False, frac=results.frac)
  
if __name__=="__main__":
  main()
