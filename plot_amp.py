#!/usr/bin/env python3

from model import *
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from functools import reduce

def config_split(config):
  ret = {}
  for i in config:
    ret[i] = {i:config[i]}
  return ret

def part_config(config,name=[]):
  if isinstance(name,str):
    print(name)
    return {name:config[name]}
  ret = {}
  for i in name:
    ret[i] = config[i]
  return ret

def hist_line(data,weights,bins,xrange=None,inter = 1):
  y,x = np.histogram(data,bins=bins,range=xrange,weights=weights)
  x = (x[:-1] + x[1:])/2
  func = interpolate.interp1d(x,y,kind="quadratic")
  delta = (xrange[1]-xrange[0])/bins/inter
  xnew = np.arange(x[0],x[-1],delta)
  ynew = func(xnew)
  return xnew,ynew

def pprint(dicts):
  s = json.dumps(dicts,indent=2)
  print(s)

params_config = {
  "m_BC":{
    "xrange":(2.15,2.65),
    "display":"$m_{ {D*}^{-}\pi^{+} }$",
    "bins":50,
    "units":"Gev",
  },
  "m_BD":{
    "xrange":(4.0,4.47),
    "display":"$m_{ {D*}^{-}{D*}^{0} }$",
    "bins":47,
    "units":"GeV"
  },
  "m_CD":{
    "xrange":(2.15,2.65),
    "display":"$m_{ {D*}^{0}\pi^{+} }$",
    "bins":50,
    "units":"GeV"
  },
  "cos_B_BD":{
    "xrange":(-1,1),
    "display":r"$\cos \theta^{ {D*}^{-} }_{ {D*}^{0} {D*}^{-} }$",
    "bins":50,
    "units":""
  }
}

for i in range(len(param_list)):
  if param_list[i] in params_config:
    params_config[param_list[i]]["idx"] = i

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

def plot(params_file="final_params.json",res_file="Resonances.json",res_list=None):
  dtype = "float32"
  w_bkg = 0.768331
  set_gpu_mem_growth()
  tf.keras.backend.set_floatx(dtype)
  with open(res_file) as f:  
    config_list = json.load(f)
  a = Model(config_list,w_bkg)
  with open(params_file) as f:  
    param = json.load(f)
  a.set_params(param)
  pprint(a.get_params())
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
  data = load_data("data")
  bg = load_data("bg")
  mcdata = load_data("PHSP")
  data_cache = a.Amp.cache_data(*data)
  bg_cache = a.Amp.cache_data(*bg)
  mcdata_cache = a.Amp.cache_data(*mcdata)
  total = a.Amp(mcdata_cache,cached=True)
  int_mc = tf.reduce_sum(total).numpy()
  n_data = data[0].shape[0]
  n_bg = bg[0].shape[0]
  norm_int = (n_data - w_bkg*n_bg)/int_mc
  a_sig = {}
  a_weight = {}
  if res_list is None:
    res_list = [[i] for i in config_list]
  config_res = [part_config(config_list,i) for i in res_list]
  fitFrac = {}
  res_name = {}
  for i in range(len(res_list)):
    name = res_list[i]
    if isinstance(name,list):
      if len(name) > 1:
        name = reduce(lambda x,y:"{}+{}".format(x,y),res_list[i])
      else :
        name = name[0]
    res_name[i] = name
    a_sig[i] = Model(config_res[i],w_bkg)
    a_sig[i].set_params(a.get_params())
    a_weight[i] = a_sig[i].Amp(mcdata_cache,cached=True).numpy()*norm_int
    fitFrac[name] = a_weight[i].sum()/(n_data - w_bkg*n_bg)
  print("FitFractions:")
  pprint(fitFrac)
  def plot_params(ax,name,bins=None,xrange=None,idx=0,display=None,units="GeV"):
    inter = 2
    if display is None: display=name
    data_hist = ax.hist(data[idx].numpy(),range=xrange,bins=bins,histtype="step",label="data")
    data_y ,data_x,_ = data_hist
    data_x = (data_x[:-1]+data_x[1:])/2
    data_err = np.sqrt(data_y)
    ax.errorbar(data_x,data_y,yerr=data_err,fmt="none")
    ax.hist(bg[idx].numpy(),range=xrange,bins=bins,histtype="step",weights=[w_bkg]*n_bg,label="bg")
    mc_bg = np.append(bg[idx].numpy(),mcdata[idx].numpy())
    mc_bg_w = np.append([w_bkg]*n_bg,total.numpy()*norm_int)
    x_mc,y_mc = hist_line(mc_bg,mc_bg_w,bins,xrange)
    ax.plot(x_mc,y_mc,label="total fit")
    for i in a_sig:
      weights = a_weight[i]
      x,y = hist_line(mcdata[idx].numpy(),weights,bins,xrange,inter)
      y = y#/2 #TODO
      ax.plot(x,y,label=res_name[i])
    ax.legend()
    ax.set_ylabel("events/({:.3f} {})".format((x_mc[1]-x_mc[0]),units))
    ax.set_xlabel("{} {}".format(display,units))
    if xrange is not None:
      ax.set_xlim(xrange[0], xrange[1])
    ax.set_ylim(0, None)
    ax.set_title(display)
  plot_list = ["m_BC","m_BD","m_CD"]
  n = len(plot_list)
  for i in range(n):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    name = plot_list[i]
    plot_params(ax,name,**params_config[name])
    fig.savefig(name)
  
  
if __name__=="__main__":
  res_list = [
    ["Zc_4025"],
    ["Zc_4160"],
    ["D1_2420","D1_2420p"],
    ["D1_2430","D1_2430p"],
    ["D2_2460","D2_2460p"],
  ]
  with tf.device("/device:CPU:0"):
    plot("final_params.json",res_list=None)
