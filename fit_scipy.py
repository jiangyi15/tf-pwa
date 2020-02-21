#!/usr/bin/env python3
from tf_pwa.model import Cache_Model,param_list,FCN
import tensorflow as tf
import time
import numpy as np
import json
import os
from scipy.optimize import minimize,BFGS,basinhopping
import tf_pwa
#from scipy.optimize import minimize,BFGS,basinhopping
from tf_pwa.angle import cal_ang_file,cal_ang_file4
from tf_pwa.utils import load_config_file,flatten_np_data,pprint,error_print,std_polar
import math

from tf_pwa.applications import fit_scipy,fit_fractions,cal_hesse_error,calPWratio,gen_data

mode = "3"
if mode=="4":
  from tf_pwa.amplitude4 import AllAmplitude4 as AllAmplitude,param_list
else:
  from tf_pwa.amplitude import AllAmplitude,param_list

from tf_pwa.cal_angle import prepare_data_from_dat_file

param_list_test = [
  "particle/A/m","particle/B/m","particle/C/m","particle/D/m","particle/BC/m", "particle/BD/m", "particle/CD/m",
  "decay/0/A->BC+D/BC/ang/beta", "decay/0/BC->B+C/B/ang/beta", "decay/0/A->BC+D/BC/ang/alpha", "decay/0/BC->B+C/B/ang/alpha",
  "decay/1/A->BD+C/BD/ang/beta", "decay/1/BD->B+D/B/ang/beta", "decay/1/A->BD+C/BD/ang/alpha", "decay/1/BD->B+D/B/ang/alpha", 
  "decay/2/A->CD+B/CD/ang/beta", "decay/2/CD->C+D/D/ang/beta", "decay/2/A->CD+B/CD/ang/alpha", "decay/2/CD->C+D/D/ang/alpha",
  "decay/1/BD->B+D/B/aligned_angle/beta","decay/0/BC->B+C/B/aligned_angle/beta",
  "decay/1/BD->B+D/D/aligned_angle/beta","decay/2/CD->C+D/D/aligned_angle/beta",
  "decay/1/BD->B+D/B/aligned_angle/alpha","decay/1/BD->B+D/B/aligned_angle/gamma",
  "decay/0/BC->B+C/B/aligned_angle/alpha","decay/0/BC->B+C/B/aligned_angle/gamma",
  "decay/1/BD->B+D/D/aligned_angle/alpha","decay/1/BD->B+D/D/aligned_angle/gamma",
  "decay/2/CD->C+D/D/aligned_angle/alpha","decay/2/CD->C+D/D/aligned_angle/gamma"
]

def load_cached_data(cached_data_file = "cached_data.npy"):
    cached_dir = "./cached_dir"
    if not os.path.exists(cached_dir):
        return None
    cached_path = os.path.join(cached_dir, cached_data_file)
    if os.path.exists(cached_path):
        cached_data = tf_pwa.load_data(cached_path)
        return cached_data
    return None

def save_cached_data(cached_data, cached_data_file = "cached_data.npy"):
    cached_dir = "./cached_dir"
    if not os.path.exists(cached_dir):
        os.mkdir(cached_dir)
    cached_path = os.path.join(cached_dir, cached_data_file)
    tf_pwa.save_data(cached_path, cached_data)

def prepare_data(dtype="float64", model="3"):
    fname = [["./data/data4600_new.dat", "data/Dst0_data4600_new.dat"],
        ["./data/bg4600_new.dat", "data/Dst0_bg4600_new.dat"],
        ["./data/PHSP4600_new.dat", "data/Dst0_PHSP4600_new.dat"]
    ]
    tname = ["data", "bg", "PHSP"]
    cached_data = load_cached_data()
    if cached_data is not None:
        data = cached_data["data"]
        bg = cached_data["bg"]
        mcdata = cached_data["PHSP"]
        print("using cached data")
        return data, bg, mcdata
    data_np = {}
    for i, name in enumerate(fname):
        data_np[tname[i]] = prepare_data_from_dat_file(name[0])
    def load_data(name):
        dat = []
        tmp = data_np[name]
        for i in param_list_test:
            tmp_data = tf.Variable(tmp[i], dtype=dtype)
            dat.append(tmp_data)
        return dat
    data, bg, mcdata = [load_data(i) for i in tname]
    save_cached_data({"data": data, "bg": bg, "PHSP": mcdata})
    return data, bg, mcdata


def prepare_data_2(dtype="float64",model="3"):
  fname = [["./data/data4600_new.dat","data/Dst0_data4600_new.dat"],
       ["./data/bg4600_new.dat","data/Dst0_bg4600_new.dat"],
       ["./data/PHSP4600_new.dat","data/Dst0_PHSP4600_new.dat"]
  ]
  tname = ["data","bg","PHSP"]
  data_np = {}
  for i in range(len(tname)):
    if model == "3" :
      data_np[tname[i]] = cal_ang_file(fname[i][0],dtype)
    elif model == "4":
      data_np[tname[i]] = cal_ang_file4(fname[i][0],fname[i][1],dtype)
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

def fit(method="BFGS",init_params="init_params.json",hesse=True,frac=True):
  POLAR = True # fit in polar coordinates. should be consistent with init_params.json if any
  GEN_TOY = False # use toy data (mcdata and bg stay the same). REMEMBER to update gen_params.json

  dtype = "float64"
  w_bkg = 0.768331
  #set_gpu_mem_growth()
  #tf.keras.backend.set_floatx(dtype)
  # open Resonances list as dict 
  config_list = load_config_file("Resonances")
  
  data, bg, mcdata = prepare_data(dtype=dtype,model=mode)
  if GEN_TOY:
    print("########## begin generate_data")
    #data = gen_data(8065,3445,w_bkg,1.1,Poisson_fluc=True)
    import pickle
    toy_file = open("toy.pkl","rb") # load pkl data
    data = pickle.load(toy_file)
    toy_file.close()
    print("########## finish generate_data")

  amp = AllAmplitude(config_list)
  a = Cache_Model(amp,w_bkg,data,mcdata,bg=bg,batch=65000)#,constrain={"Zc_4160_g0:0":(0.1,0.1)})
  if POLAR:
    print("Fitting parameters are defined in POLAR coordinates")
  else:
    print("Fitting parameters are defined in XY coordinates")
  #print(type(a.Amp))
  try :
    with open(init_params) as f:  
      param = json.load(f)
      print("using {}".format(init_params))
      if "config" in param:
        amp = AllAmplitude(param["config"])
        a = Cache_Model(amp,w_bkg,data,mcdata,bg=bg,batch=65000)
      if "value" in param:
        a.set_params(param["value"])
      else :
        a.set_params(param)
  except Exception as e:
    #print(e)
    print("using RANDOM parameters")
  amp.trans_params(polar=POLAR)
  #print(a.Amp(data))
  #exit()
  #a.Amp.polar=POLAR

  bounds_dict = {
      "Zc_4160_m":(4.1,4.22),
      "Zc_4160_g":(0,None),
      #"D1_2420r": (3.12,10.0)
  }

  #args = a.Amp.get_all_dic(trainable_only=True)
  args_name = a.Amp.trainable_vars
  
  pprint(a.get_params())
  #print(data,bg,mcdata)
  #t = time.time()
  #nll,g = a.cal_nll_gradient()#data_w,mcdata,weight=weights,batch=50000)
  #print("nll:",nll,"Time:",time.time()-t)
  #exit()
  print("########## chain decay:")
  for i in a.Amp.A.chain_decay():
    print(i)
  now = time.time()
  s, nlls, points = fit_scipy(model=a,bounds_dict=bounds_dict)

  print("########## fit state:")
  print(s)
  print("\nTime for fitting:",time.time()-now)
  
  val = dict(zip(args_name,s.x))
  a.set_params(val)
  params = a.get_params()
  with open("fit_curve.json","w") as f:
    json.dump({"nlls": nlls, "points": points},f,indent=2)
  err=None
  if hesse:
    inv_he = cal_hesse_error(a)
    diag_he = inv_he.diagonal()
    hesse_error = np.sqrt(diag_he).tolist()
    err = dict(zip(args_name,hesse_error))
  print("\n########## fit results:")
  for i in val:
    if hesse:
      print("  ",i,":",error_print(val[i],err[i]))
    else:
      print("  ",i,":",val[i])

  '''    
  print("\n########## fitting params in polar expression")
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
  #for v in config_list:
    #rho = params[v.rstrip('pm')+'r:0']
    #phi = params[v+'i:0']
    #rho,phi = std_polar(rho,phi)
    #params[v.rstrip('pm')+'r:0'] = rho
    #params[v+'i:0'] = phi
    #print(v,"\t\t%.5f * exp(%.5fi)"%(rho,phi))
  #a.set_params(params)
  
  #with a.Amp.params_form(polar=True) as params:
    #pprint(params)'''

  outdic={"value":params,"error":err,"config":config_list}
  with open("final_params.json","w") as f:                                      
    json.dump(outdic,f,indent=2)
  #print("\n########## ratios of partial wave amplitude square")
  #calPWratio(params,POLAR)
  
  if frac:
    frac, err_frac = fit_fractions(a,mcdata,config_list,inv_he,hesse)
    print("########## fit fractions")
    for i in config_list:
      print(i,":",error_print(frac[i],err_frac[i]))
  print("\nEND\n")
  #return frac,config_list,params

def main():
  import argparse
  parser = argparse.ArgumentParser(description="simple fit scripts")
  parser.add_argument("--no-hesse", action="store_false", default=True,dest="hesse")
  parser.add_argument("--no-frac", action="store_false", default=True,dest="frac")
  parser.add_argument("--no-GPU", action="store_false", default=False,dest="has_gpu")
  parser.add_argument("--method", default="BFGS",dest="method")
  results = parser.parse_args()
  if results.has_gpu:
    with tf.device("/device:GPU:0"):
      fit(method=results.method, hesse=results.hesse, frac=results.frac)
  else:
    with tf.device("/device:CPU:0"):
      fit(method=results.method, hesse=results.hesse, frac=results.frac)

  '''frac_list = {}
  params_list = {}
  frac,config_list,params=fit(method=results.method, hesse=False, frac=results.frac)
  for reson in config_list:
    frac_list[reson]=[frac[reson]]
  for p in params:
    params_list[p] = [params[p]]
  for i in range(100):
    frac,c,params=fit(method=results.method, hesse=False, frac=results.frac)
    for reson in config_list:
      frac_list[reson].append(frac[reson])
    for p in params:
      params_list[p].append(params[p])
  for reson in config_list:
    print(reson+"=",frac_list[reson])
  for p in params_list:
    print(p[:-2]+"=",params_list[p])'''

if __name__ == "__main__":
  main()
