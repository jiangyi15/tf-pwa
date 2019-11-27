#!/usr/bin/env python3

from model import *
import json
import numpy as np
import matplotlib.pyplot as plt

def config_split(config):
  ret = {}
  for i in config:
    ret[i] = {i:config[i]}
  return ret

def hist_line(data,weights,bins,range=None):
  y,x = np.histogram(data,bins=bins,range=range,weights=weights)
  x = (x[:-1] + x[1:])/2
  return x,y

def pprint(dicts):
  s = json.dumps(dicts,indent=2)
  print(s)

params_config = {
  "m_BC":{
    "range":(2.15,2.65),
    "display":"$m_{ {D*}^{-}\pi^{+} }$",
    "bins":50,
    "units":"Gev",
  },
  "m_BD":{
    "range":(4.0,4.47),
    "display":"$m_{ {D*}^{-}{D*}^{0} }$",
    "bins":47,
    "units":"GeV"
  },
  "m_CD":{
    "range":(2.15,2.65),
    "display":"$m_{ {D*}^{0}\pi^{+} }$",
    "bins":50,
    "units":"GeV"
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
  def plot_params(name,bins=None,range=None,idx=0,display=None,units="GeV"):
    if display is None: display=name
    plt.clf()
    plt.hist(data[idx].numpy(),range=range,bins=bins,histtype="step",label="data")
    plt.hist(bg[idx].numpy(),range=range,bins=bins,histtype="step",weights=[w_bkg]*n_bg,label="bg")
    mc_bg = np.append(bg[idx].numpy(),mcdata[idx].numpy())
    mc_bg_w = np.append([w_bkg]*n_bg,total.numpy()*norm_int)
    x_mc,y_mc = hist_line(mc_bg,mc_bg_w,bins,range)
    plt.plot(x_mc,y_mc,label="total fit")
    for i in config_res:
      weights = a_weight[i]
      x,y = hist_line(mcdata[idx].numpy(),weights,bins,range)
      y = y#/2 #TODO
      plt.plot(x,y,label=i)
    plt.legend()
    plt.ylabel("events/({:.3f} {})".format((x_mc[1]-x_mc[0]),units))
    plt.xlabel("{} {}".format(display,units))
    plt.title(display)
    plt.savefig(name)
  for i in params_config:
    plot_params(i,**params_config[i])
  
  
if __name__=="__main__":
  main()
