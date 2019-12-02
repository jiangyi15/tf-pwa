#!/usr/bin/env python3

from model import *
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def config_split(config):
  ret = {}
  for i in config:
    ret[i] = {i:config[i]}
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

def main(params_file="final_params.json"):
  
  dtype = "float32"
  w_bkg = 0.768331
  set_gpu_mem_growth()
  for i in range(len(param_list)):
    if param_list[i] in params_config:
      params_config[param_list[i]]["idx"] = i
  tf.keras.backend.set_floatx(dtype)
  with open("Resonances.json") as f:  
    config_list = json.load(f)
  a = Model(config_list,w_bkg)
  with open(params_file) as f:  
    param = json.load(f)
  a.set_params(param)
  
  pprint(a.get_params())
  def load_data(fname):
    dat = []
    with open(fname) as f:
      tmp = json.load(f)
      for i in param_list:
        tmp_data = tf.Variable(tmp[i],name=i,dtype=dtype)
        dat.append(tmp_data)
    return dat
  data = load_data("./data/data_ang_n4.json")
  bg = load_data("./data/bg_ang_n4.json")
  mcdata = load_data("./data/PHSP_ang_n4.json")
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
  config_res = config_split(config_list)
  fitFrac = {}
  for i in config_res:
    a_sig[i] = Model(config_res[i],w_bkg)
    a_sig[i].set_params(a.get_params())
    a_weight[i] = a_sig[i].Amp(mcdata_cache,cached=True).numpy()*norm_int
    fitFrac[i] = a_weight[i].sum()/(n_data - w_bkg*n_bg)
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
    for i in config_res:
      weights = a_weight[i]
      x,y = hist_line(mcdata[idx].numpy(),weights,bins,xrange,inter)
      y = y#/2 #TODO
      ax.plot(x,y,label=i)
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
  with tf.device("/device:CPU:0"):
    main()
