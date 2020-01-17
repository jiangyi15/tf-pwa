#!/usr/bin/env python3

from tf_pwa.model import *
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from functools import reduce
from tf_pwa.angle import cal_ang_file,EularAngle
from tf_pwa.utils import load_config_file,flatten_np_data,pprint
import os
from math import pi
from generate_toy import generate_data

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
    if i in config:
      ret[i] = config[i]
  return ret

def equal_pm(a,b):
  def remove_pm(s):
    ret = s
    if s.endswith("p"):
      ret = s[:-1]
    elif s.endswith("m"):
      ret = s[:-1]
    return ret
  return remove_pm(a) == remove_pm(b)
  

def part_combine_pm(config_list):
  ret = []
  for i in config_list:
    for j in ret:
      if equal_pm(i,j[0]):
        j.append(i)
        break
    else:
      ret.append([i])
  return ret

def hist_line(data,weights,bins,xrange=None,inter = 1):
  y,x = np.histogram(data,bins=bins,range=xrange,weights=weights)
  x = (x[:-1] + x[1:])/2
  func = interpolate.interp1d(x,y,kind="quadratic")
  delta = (xrange[1]-xrange[0])/bins/inter
  xnew = np.arange(x[0],x[-1],delta)
  ynew = func(xnew)
  return xnew,ynew


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
  "alpha_B_BD":{
    "xrange":(-pi,pi),
    "display":r"$\phi^{ {D*}^{-} }_{ {D*}^{0} {D*}^{-} }$",
    "bins":50,
    "units":"",
    "legend":False
  },
  "alpha_BD":{
    "xrange":(-pi,pi),
    "display":r"$ \phi_{ {D*}^{0} {D*}^{-} }$",
    "bins":50,
    "units":"",
    "legend":False
  },
  "cosbeta_BD":{
    "xrange":(-1,1),
    "display":r"$\cos \theta_{ {D*}^{0} {D*}^{-} }$",
    "bins":50,
    "units":"",
    "legend":False
  },
  "cosbeta_B_BD":{
    "xrange":(-1,1),
    "display":r"$\cos \theta^{ {D*}^{-} }_{ {D*}^{0} {D*}^{-} }$",
    "bins":50,
    "units":"",
    "legend":False
  },
  "alpha_B_BC":{
    "xrange":(-pi,pi),
    "display":r"$\phi^{ {D*}^{-} }_{ {D*}^{-}\pi^{+} }$",
    "bins":50,
    "units":"",
    "legend":False
  },
  "alpha_BC":{
    "xrange":(-pi,pi),
    "display":r"$ \phi_{ {D*}^{-} \pi^{+} }$",
    "bins":50,
    "units":"",
    "legend":False
  },
  "cosbeta_BC":{
    "xrange":(-1,1),
    "display":r"$\cos \theta_{ {D*}^{-} \pi^{+} }$",
    "bins":50,
    "units":"",
    "legend":False
  },
  "cosbeta_B_BC":{
    "xrange":(-1,1),
    "display":r"$\cos \theta^{ {D*}^{-} }_{ {D*}^{-}\pi^{+} }$",
    "bins":50,
    "units":"",
    "legend":False
  },
  "alpha_D_CD":{
    "xrange":(-pi,pi),
    "display":r"$\phi^{ {D*}^{0} }_{ {D*}^{0}\pi^{+} }$",
    "bins":50,
    "units":"",
    "legend":False
  },
  "alpha_CD":{
    "xrange":(-pi,pi),
    "display":r"$ \phi_{ {D*}^{0} \pi^{+} }$",
    "bins":50,
    "units":"",
    "legend":False
  },
  "cosbeta_CD":{
    "xrange":(-1,1),
    "display":r"$\cos \theta_{ {D*}^{0} \pi^{+} }$",
    "bins":50,
    "units":"",
    "legend":False
  },
  "cosbeta_D_CD":{
    "xrange":(-1,1),
    "display":r"$\cos \theta^{ {D*}^{0} }_{ {D*}^{0}\pi^{+} }$",
    "bins":50,
    "units":"",
    "legend":False
  },
  
}

model = 3

for i in range(len(param_list)):
  name = param_list[i]
  if name.startswith("beta"):
    name = "cos" + name
  if name in params_config:
    params_config[name]["idx"] = i

def prepare_data(dtype="float64",model="3"):
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

def plot(params_file="final_params.json",res_file="Resonances",res_list=None,pm_combine=True):
  POLAR=True
  dtype = "float64"
  w_bkg = 0.768331
  set_gpu_mem_growth()
  tf.keras.backend.set_floatx(dtype)
  config_list = config_list = load_config_file(res_file)
  a = Model(config_list,w_bkg,kwargs={"polar":POLAR})
  #a.Amp.polar=POLAR
  with open(params_file) as f:  
    param = json.load(f)
    if "value" in param:
      a.set_params(param["value"])
    else :
      a.set_params(param)
  pprint(a.get_params())
  
  data, bg, mcdata = prepare_data()
  #data = generate_data(8065,3445,w_bkg,1.1,True)
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
    if pm_combine:
      res_list = part_combine_pm(config_list)
    else:
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
    a_sig[i] = Model(config_res[i],w_bkg,kwargs={"polar":POLAR})
    #a_sig[i].Amp.polar=POLAR
    a_sig[i].set_params(a.get_params())
    a_weight[i] = a_sig[i].Amp(mcdata_cache,cached=True).numpy()*norm_int
    fitFrac[name] = a_weight[i].sum()/(n_data - w_bkg*n_bg)
  print("FitFractions:")
  pprint(fitFrac)
  
  cmap = plt.get_cmap("jet")
  N = 3 + len(res_list)
  colors = [cmap(float(i)/N) for i in range(N)]
  colors = [
    "black","red","green","blue","yellow","magenta","cyan","purple","teal","springgreen","azure"
  ] + colors
  
  def plot_params(ax,name,bins=None,xrange=None,idx=0,display=None,units="GeV",legend=True):
    fd = lambda x:x
    if name.startswith("cos"):
      fd = lambda x:np.cos(x)
    inter = 2
    color = iter(colors)
    if display is None: display=name
    data_hist = np.histogram(fd(data[idx].numpy()),range=xrange,bins=bins)
    #ax.hist(fd(data[idx].numpy()),range=xrange,bins=bins,histtype="step",label="data",zorder=99,color="black")
    data_y ,data_x = data_hist[0:2]
    data_x = (data_x[:-1]+data_x[1:])/2
    data_err = np.sqrt(data_y)
    ax.errorbar(data_x,data_y,yerr=data_err,fmt=".",color=next(color),zorder = -2)
    if bg is not None:
      ax.hist(fd(bg[idx].numpy()),range=xrange,bins=bins,histtype="stepfilled",alpha=0.5,color="grey",weights=[w_bkg]*n_bg,label="bg",zorder = -1)
      mc_bg = fd(np.append(bg[idx].numpy(),mcdata[idx].numpy()))
      mc_bg_w = np.append([w_bkg]*n_bg,total.numpy()*norm_int)
    else:
      mc_bg = fd(mcdata[idx].numpy())
      mc_bg_w = total.numpy()*norm_int
    x_mc,y_mc = hist_line(mc_bg,mc_bg_w,bins,xrange)
    #ax.plot(x_mc,y_mc,label="total fit")
    ax.hist(mc_bg,weights=mc_bg_w,bins=bins,range=xrange,histtype="step",color=next(color),label="total fit",zorder = 100)
    for i in a_sig:
      weights = a_weight[i]
      x,y = hist_line(fd(mcdata[idx].numpy()),weights,bins,xrange,inter)
      y = y
      ax.plot(x,y,label=res_name[i],linestyle="solid",linewidth=1,color=next(color))
    if legend:
      ax.legend(framealpha=0.5,fontsize="small")
    ax.set_ylabel("events/({:.3f} {})".format((x_mc[1]-x_mc[0]),units))
    ax.set_xlabel("{} {}".format(display,units))
    if xrange is not None:
      ax.set_xlim(xrange[0], xrange[1])
    ax.set_ylim(0, None)
    ax.set_title(display)
  plot_list = [
    "m_BC","m_BD","m_CD",
    "alpha_BD","cosbeta_BD","alpha_B_BD","cosbeta_B_BD",
    "alpha_BC","cosbeta_BC","alpha_B_BC","cosbeta_B_BC",
    "alpha_CD","cosbeta_CD","alpha_D_CD","cosbeta_D_CD"
  ]
  n = len(plot_list)
  if not os.path.exists("figure"):
    os.mkdir("figure")
  #plt.style.use("classic") 
  for i in range(n):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    name = plot_list[i]
    plot_params(ax,name,**params_config[name])
    fig.savefig("figure/"+name+".pdf")
    fig.savefig("figure/"+name+".png",dpi=300)
 


def calPWratio(params,POLAR=True):
  dtype = "float64"
  w_bkg = 0.768331
  set_gpu_mem_growth()
  tf.keras.backend.set_floatx(dtype)
  config_list = load_config_file("Resonances")
  a = Model(config_list,w_bkg,kwargs={"polar":POLAR})
  
  args_name = []
  for i in a.Amp.trainable_variables:
    args_name.append(i.name)
  #a.Amp.polar=True

  a.set_params(params)
  if not POLAR:# if final_params.json is not in polar coordinates
    i = 0 
    for v in args_name:
      if len(v)>15:
        if i%2==0:
          tmp_name = v
          tmp_val = params[v]
        else:
          params[tmp_name] = np.sqrt(tmp_val**2+params[v]**2)
          params[v] = np.arctan2(params[v],tmp_val)
      i+=1
    a.set_params(params)
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
  a_sig = {}

  #res_list = [[i] for i in config_list]
  res_list = [
    ["Zc_4025"],
    ["Zc_4160"],
    ["D1_2420","D1_2420p"],
    ["D1_2430","D1_2430p"],
    ["D2_2460","D2_2460p"],
  ]
  
  config_res = [part_config(config_list,i) for i in res_list]
  PWamp = {}
  for i in range(len(res_list)):
    name = res_list[i]
    if isinstance(name,list):
      if len(name) > 1:
        name = reduce(lambda x,y:"{}+{}".format(x,y),res_list[i])
      else :
        name = name[0]
    a_sig[i] = Model(config_res[i],w_bkg,kwargs={"polar":POLAR})
    p_list = [[],[]]
    for p in a_sig[i].get_params():
      if p[-3]=='r' and len(p)>15:
        if p[8]=='d':
          p_list[1].append(p)
        else:
          p_list[0].append(p)
    first = True
    for p in p_list[0]:
      a_sig[i].set_params(params)
      for q in p_list[0]:
        a_sig[i].set_params({q:0})
      a_sig[i].set_params({p:params[p]})
      if first:
        norm = a_sig[i].Amp(mcdata_cache,cached=True).numpy().sum()
        print(p[:-3],"\t",1.0)
        first = False
      else:
        print(p[:-3],"\t",a_sig[i].Amp(mcdata_cache,cached=True).numpy().sum()/norm)
    first = True
    for p in p_list[1]:
      a_sig[i].set_params(params)
      for q in p_list[1]:
        a_sig[i].set_params({q:0})
      a_sig[i].set_params({p:params[p]})
      if first:
        norm = a_sig[i].Amp(mcdata_cache,cached=True).numpy().sum()
        print(p[:-3],"\t",1.0)
        first = False
      else:
        print(p[:-3],"\t",a_sig[i].Amp(mcdata_cache,cached=True).numpy().sum()/norm)
    print()
    #print(a_sig[i].get_params())
    #a_weight[i] = a_sig[i].Amp(mcdata_cache,cached=True).numpy()
    #PWamp[name] = a_weight[i].sum()/(n_data - w_bkg*n_bg)


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

  #with open("final_params.json") as f:  
  #  params = json.load(f)
  #  params = params["value"]
  #calPWratio(params,POLAR=True)
