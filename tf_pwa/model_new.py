import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from .amplitude_new import AllAmplitude


class Model:
  def __init__(self,res,w_bkg = 0):
    self.Amp = AllAmplitude(res)
    self.w_bkg = w_bkg
    
  def nll(self,data,bg,mcdata):
    ln_data = tf.reduce_sum(tf.math.log(self.Amp(data)))
    ln_bg = tf.reduce_sum(tf.math.log(self.Amp(bg)))
    int_mc = tf.math.log(tf.reduce_mean(self.Amp(mcdata)))
    n_data = data[0].shape[0]
    n_bg = bg[0].shape[0]
    n_mc = mcdata[0].shape[0]
    return -(ln_data - self.w_bkg * ln_bg - (n_data - self.w_bkg*n_bg) * int_mc)
  

param_list = [
  "m_BC","phi_BC","cos_BC","phi_B_BC","cos_B_BC",
	"alpha_B_BC","cosbeta_B_BC","gamma_B_BC",
	"m_BD","phi_BD","cos_BD","phi_B_BD","cos_B_BD",
	"alpha_B_BD","cosbeta_B_BD","gamma_B_BD",
	"alpha_D_BD","cosbeta_D_BD","gamma_D_BD",
	"m_CD","phi_CD","cos_CD","phi_D_CD","cos_D_CD",
	"alpha_D_CD","cosbeta_D_CD","gamma_D_CD"
]
              
config_list = {"D2_2460"
    :{
        "m0":2.4607,
        "m_min":2.4603,
        "m_max":2.4611,
        "g0":0.0475,
        "g_min":0.0464,
        "g_max":0.0486,
        "J":2,
        "Par":1,
        "Chain":21
    },
    "D2_2460p"
    :{
        "m0":2.4654,
        "m_min":2.4644,
        "m_max":2.4667,
        "g0":0.0467,
        "g_min":0.0455,
        "g_max":0.0479,
        "J":2,
        "Par":1,
        "Chain":121
    },
    "D1_2430"
    :{
        "m0":2.427,
        "m_min":2.387,
        "m_max":2.467,
        "g0":0.284,
        "g_min":0.274,
        "g_max":0.514,
        "J":1,
        "Par":1,
        "Chain":12
    },
    "D1_2430p"
    :{
        "m0":2.427,
        "m_min":2.387,
        "m_max":2.467,
        "g0":0.384,
        "g_min":0.274,
        "g_max":0.514,
        "J":1,
        "Par":1,
        "Chain":112
    },
    "D1_2420"
    :{
        "m0":2.4208,
        "m_min":2.4203,
        "m_max":2.4213,
        "g0":0.0317,
        "g_min":0.0292,
        "g_max":0.0342,
        "J":1,
        "Par":1,
        "Chain":11
    },
    "D1_2420p"
    :{
        "m0":2.4232,
        "m_min":2.4208,
        "m_max":2.4256,
        "g0":0.025,
        "g_min":0.019,
        "g_max":0.021,
        "J":1,
        "Par":1,
        "Chain":111
    },
    "Zc_4025"
    :{
        "m0":4.0263,
        "g0":0.0248,
        "J":1,
        "Par":1,
        "Chain":-1
    },
    "Zc_4160"
    :{
        "m0":4.1628,
        "g0":0.0701,
        "J":1,
        "Par":1,
        "Chain":-2
    }
}

def train_one_step(model, optimizer, data, bg,mc):
  with tf.GradientTape() as tape:
    loss = model.nll(data,bg,mc)

  grads = tape.gradient(loss, model.Amp.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.Amp.trainable_variables))
  print("loss:",loss)
  return loss

import time as t

def main():
  import json
  a = Model(config_list,0.8)
  data = []
  bg = []
  mcdata = []
  with open("./data/PHSP_ang_n.json") as f:
    tmp = json.load(f)
    for i in param_list:
      tmp_data = tf.Variable(tmp[i],name=i)
      mcdata.append(tmp_data)
  with open("./data/data_ang_n.json") as f:
    tmp = json.load(f)
    for i in param_list:
      tmp_data = tf.Variable(tmp[i],name=i)
      data.append(tmp_data)
  with open("./data/bg_ang_n.json") as f:
    tmp = json.load(f)
    for i in param_list:
      tmp_data = tf.Variable(tmp[i],name=i)
      bg.append(tmp_data)
  #print("data,bg,mcdata",data,bg,mcdata)
  data_set = tf.data.Dataset.from_tensor_slices(tuple(data))
  data_set = data_set.shuffle(10000).batch(800)
  data_set_it = iter(data_set)
  bg_set = tf.data.Dataset.from_tensor_slices(tuple(bg))
  bg_set = bg_set.shuffle(10000).batch(340)
  bg_set_it = iter(bg_set)
  mc_set = tf.data.Dataset.from_tensor_slices(tuple(mcdata))
  mc_set = mc_set.shuffle(10000).batch(2520)
  mc_set_it = iter(mc_set)
  t1 = t.time()
  with tf.device('/device:GPU:0'):
    print("NLL0:",a.nll(data,bg,mcdata))#.collect_params())
  optimizer = tf.keras.optimizers.Adagrad()
  t2 = t.time()
  for i in range(50):
    try :
      data_i = data_set_it.get_next()
      bg_i = bg_set_it.get_next()
      mcdata_i = mc_set_it.get_next()
      print(i,end=': ')
      train_one_step(a,optimizer,data_i,bg_i,mcdata_i);
    except:
      data_set = tf.data.Dataset.from_tensor_slices(tuple(data))
      data_set = data_set.shuffle(10000).batch(800)
      data_set_it = iter(data_set)
      bg_set = tf.data.Dataset.from_tensor_slices(tuple(bg))
      bg_set = bg_set.shuffle(10000).batch(340)
      bg_set_it = iter(bg_set)
      mc_set = tf.data.Dataset.from_tensor_slices(tuple(mcdata))
      mc_set = mc_set.shuffle(10000).batch(2520)
      mc_set_it = iter(mc_set)
  print("i",i)    
  #now = time.time()
  #with tf.device('/device:GPU:0'):
    #print(a(x))#.collect_params())
  #print(time.time()-now)
  t3 = t.time()
  with tf.device('/device:GPU:0'):
    print("NLL:",a.nll(data,bg,mcdata))#.collect_params())
  t4 = t.time()
  print("Time:",t2-t1,t3-t2,t4-t3)
  print("Variables",a.Amp.trainable_variables)

  ### plot
  ndata = 8065
  nbg = 3445
  bgw = 0.8
  w = a.Amp(mcdata)
  weight = w.numpy()

  n_reson = 8
  reson_1 = "D2_2460"
  reson_2 = "D2_2460p"
  reson_3 = "D1_2430"
  reson_4 = "D1_2430p"
  reson_5 = "D1_2420"
  reson_6 = "D1_2420p"
  reson_7 = "Zc_4025"
  reson_8 = "Zc_4160"
  reson_variables = []

  for i in range(1,n_reson+1):
      exec("config_list_%s = {reson_%s:config_list[reson_%s]}" % (i,i,i))
      locals()["a_%s"%i] = Model(locals()["config_list_%s"%i],0.8)
      for j in locals()["a_%s"%i].Amp.trainable_variables:
          reson_variables.append(j)

  for v in a.Amp.trainable_variables:
      for u in reson_variables:
          if u.name == v.name:
              u.assign(v)

  # for m_BC variable
  var_name,var_num = "m_BC",0
  locals()[var_name] = mcdata[var_num].numpy()
  locals()[var_name+"_data"] = data[var_num].numpy()
  locals()[var_name+"_bg"] = bg[var_num].numpy()
  xbinmin,xbinmax = 2.15,2.65

  nn = plt.hist(locals()[var_name],bins=50,weights=weight,range=(xbinmin,xbinmax))
  plt.title("Total "+var_name)
  plt.clf()
  nmcwei = sum(nn[0])

  for i in range(1,n_reson+1):
      locals()["w_%s"%i] = locals()["a_%s"%i].Amp(mcdata)
      locals()["weight_%s"%i] = locals()["w_%s"%i].numpy()
      locals()["nn_%s"%i] = plt.hist(locals()[var_name],bins=50,weights=locals()["weight_%s"%i],range=(xbinmin,xbinmax))
      plt.title(locals()["reson_%s"%i])
      plt.savefig("fig/"+locals()["reson_%s"%i]+"_"+var_name)
      plt.clf()
      
  xbin = []
  for i in range(50):
      xbin.append((nn[1][i+1]+nn[1][i])/2)
      
  plt.hist(locals()[var_name+"_data"],bins=50,range=(xbinmin,xbinmax))
  (counts, bins) = np.histogram(locals()[var_name+"_bg"],bins=50,range=(xbinmin,xbinmax))
  plt.hist(bins[:-1],bins,weights=bgw*counts)
  ybin = nn[0]*(ndata-bgw*nbg)/nmcwei + bgw*counts
  plt.plot(xbin,ybin,label="total")
  plt.title(var_name)

  for i in range(1,n_reson+1):
      locals()["ybin_%s"%i] = locals()["nn_%s"%i][0]*(ndata-bgw*nbg)/nmcwei
      plt.plot(xbin,locals()["ybin_%s"%i],label=locals()["reson_%s"%i])

  plt.legend()
  plt.savefig("fig/"+var_name)    
  plt.clf()

  # for m_BD variable
  var_name,var_num = "m_BD",8
  locals()[var_name] = mcdata[var_num].numpy()
  locals()[var_name+"_data"] = data[var_num].numpy()
  locals()[var_name+"_bg"] = bg[var_num].numpy()
  xbinmin,xbinmax = 4,4.5

  nn = plt.hist(locals()[var_name],bins=50,weights=weight,range=(xbinmin,xbinmax))
  plt.title("Total "+var_name)
  plt.clf()
  nmcwei = sum(nn[0])

  for i in range(1,n_reson+1):
      locals()["w_%s"%i] = locals()["a_%s"%i].Amp(mcdata)
      locals()["weight_%s"%i] = locals()["w_%s"%i].numpy()
      locals()["nn_%s"%i] = plt.hist(locals()[var_name],bins=50,weights=locals()["weight_%s"%i],range=(xbinmin,xbinmax))
      plt.title(locals()["reson_%s"%i])
      plt.savefig("fig/"+locals()["reson_%s"%i]+"_"+var_name)
      plt.clf()
      
  xbin = []
  for i in range(50):
      xbin.append((nn[1][i+1]+nn[1][i])/2)
      
  plt.hist(locals()[var_name+"_data"],bins=50,range=(xbinmin,xbinmax))
  (counts, bins) = np.histogram(locals()[var_name+"_bg"],bins=50,range=(xbinmin,xbinmax))
  plt.hist(bins[:-1],bins,weights=bgw*counts)
  ybin = nn[0]*(ndata-bgw*nbg)/nmcwei + bgw*counts
  plt.plot(xbin,ybin,label="total")
  plt.title(var_name)

  for i in range(1,n_reson+1):
      locals()["ybin_%s"%i] = locals()["nn_%s"%i][0]*(ndata-bgw*nbg)/nmcwei
      plt.plot(xbin,locals()["ybin_%s"%i],label=locals()["reson_%s"%i])

  plt.legend()
  plt.savefig("fig/"+var_name)    
  plt.clf()

  # for m_CD variable
  var_name,var_num = "m_CD",19
  locals()[var_name] = mcdata[var_num].numpy()
  locals()[var_name+"_data"] = data[var_num].numpy()
  locals()[var_name+"_bg"] = bg[var_num].numpy()
  xbinmin,xbinmax = 2.15,2.65

  nn = plt.hist(locals()[var_name],bins=50,weights=weight,range=(xbinmin,xbinmax))
  plt.title("Total "+var_name)
  plt.clf()
  nmcwei = sum(nn[0])

  for i in range(1,n_reson+1):
      locals()["w_%s"%i] = locals()["a_%s"%i].Amp(mcdata)
      locals()["weight_%s"%i] = locals()["w_%s"%i].numpy()
      locals()["nn_%s"%i] = plt.hist(locals()[var_name],bins=50,weights=locals()["weight_%s"%i],range=(xbinmin,xbinmax))
      plt.title(locals()["reson_%s"%i])
      plt.savefig("fig/"+locals()["reson_%s"%i]+"_"+var_name)
      plt.clf()
      
  xbin = []
  for i in range(50):
      xbin.append((nn[1][i+1]+nn[1][i])/2)
      
  plt.hist(locals()[var_name+"_data"],bins=50,range=(xbinmin,xbinmax))
  (counts, bins) = np.histogram(locals()[var_name+"_bg"],bins=50,range=(xbinmin,xbinmax))
  plt.hist(bins[:-1],bins,weights=bgw*counts)
  ybin = nn[0]*(ndata-bgw*nbg)/nmcwei + bgw*counts
  plt.plot(xbin,ybin,label="total")
  plt.title(var_name)

  for i in range(1,n_reson+1):
      locals()["ybin_%s"%i] = locals()["nn_%s"%i][0]*(ndata-bgw*nbg)/nmcwei
      plt.plot(xbin,locals()["ybin_%s"%i],label=locals()["reson_%s"%i])

  plt.legend()
  plt.savefig("fig/"+var_name)    
  plt.clf()

  '''# for cosTheta_BC variable
  var_name,var_num = "cosTheta_BC",3
  locals()[var_name] = mcdata[var_num].numpy()
  locals()[var_name+"_data"] = data[var_num].numpy()
  locals()[var_name+"_bg"] = bg[var_num].numpy()
  xbinmin,xbinmax = -1,1

  nn = plt.hist(locals()[var_name],bins=50,weights=weight,range=(xbinmin,xbinmax))
  plt.title("Total "+var_name)
  plt.clf()
  nmcwei = sum(nn[0])

  for i in range(1,n_reson+1):
      locals()["w_%s"%i] = locals()["a_%s"%i].Amp(mcdata)
      locals()["weight_%s"%i] = locals()["w_%s"%i].numpy()
      locals()["nn_%s"%i] = plt.hist(locals()[var_name],bins=50,weights=locals()["weight_%s"%i],range=(xbinmin,xbinmax))
      plt.title(locals()["reson_%s"%i])
      plt.clf()
      
  xbin = []
  for i in range(50):
      xbin.append((nn[1][i+1]+nn[1][i])/2)
      
  plt.hist(locals()[var_name+"_data"],bins=50,range=(xbinmin,xbinmax))
  (counts, bins) = np.histogram(locals()[var_name+"_bg"],bins=50,range=(xbinmin,xbinmax))
  plt.hist(bins[:-1],bins,weights=bgw*counts)
  ybin = nn[0]*(ndata-bgw*nbg)/nmcwei + bgw*counts
  plt.plot(xbin,ybin)
  plt.title(var_name)

  for i in range(1,n_reson+1):
      locals()["ybin_%s"%i] = locals()["nn_%s"%i][0]*(ndata-bgw*nbg)/nmcwei
      plt.plot(xbin,locals()["ybin_%s"%i])

  plt.savefig(var_name)    
  plt.clf() '''

### main
if __name__=="__main__":
  t0=t.time()
  print("Start-----")
  main()
  print("Elasped Time:",t.time()-t0)

