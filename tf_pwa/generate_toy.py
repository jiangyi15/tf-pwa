import tensorflow as tf
import numpy as np
import json
from .model import *
from .angle import cal_ang_file,cal_ang_file4
from .utils import load_config_file,flatten_np_data

def prepare_data(fname,dtype="float64"):
  data_np = cal_ang_file(fname,dtype)
  dat = []
  tmp = flatten_np_data(data_np)
  for i in param_list:
    tmp_data = tf.Variable(tmp[i],name=i,dtype=dtype)
    dat.append(tmp_data)
  return dat

def generate_data(Ndata,Nbg,wbg,scale,Poisson_fluc=False):
  POLAR = True # depends on whether gen_params.json is defined in polar coordinates
  Nbg = round(wbg*Nbg)
  Nmc = Ndata-Nbg #8065-3445*0.768331
  if Poisson_fluc:#Poisson
    Nmc = np.random.poisson(Nmc)
    Nbg = np.random.poisson(Nbg)
  print("data:",Nmc+Nbg,", sig:",Nmc,", bkg:",Nbg)
  dtype = "float64"
  #set_gpu_mem_growth()
  tf.keras.backend.set_floatx(dtype)
  config_list = load_config_file("Resonances")
  phsp = prepare_data("./data/PHSP4600_new.dat",dtype)
  batch = 65000
  
  amp = AllAmplitude(config_list)
  amp.polar=POLAR
  with open("gen_params.json") as f:
    param = json.load(f)
  param = param["value"]
  amp.set_params(param)
  mcdata = amp.cache_data(*phsp,batch=batch) #mcdata is the cached phsp
  ampsq = []
  for i in range(len(mcdata)):
    ampsq.append(amp(mcdata[i],cached=True))
  ampsq = tf.concat(ampsq,axis=0)
  ampsq_max = tf.reduce_max(ampsq).numpy()
  ampsq_len = ampsq.__len__()
  Nsample = round(Nmc*scale*10)
  if Nsample>ampsq_len:
    raise Exception("Not enough input PHSP sample")
  uni_rdm = tf.random.uniform([Nsample],minval=0,maxval=scale*ampsq_max,dtype=dtype)

  data_tmp = []
  phsp = tf.transpose(phsp)
  list_rdm = tf.random.uniform([Nsample],dtype=tf.int64,maxval=ampsq_len)
  n = 0
  j = 0
  for i in list_rdm:
    if ampsq[i]>uni_rdm[j]:
      data_tmp.append(phsp[i])
      n+=1
    j+=1
    if n==Nmc:
      break
  if n<Nmc:
    raise Exception("Not enough MC")
  data_tmp = tf.stack(data_tmp)

  bg = prepare_data("./data/bg4600_new.dat",dtype)
  bg = tf.transpose(bg)
  bg_idx = tf.random.uniform([Nbg],dtype=tf.int64,maxval=len(bg))#np.random.randint(len(bg),size=Nbg)
  bg_tmp = []
  n = 0
  for i in bg_idx:
    bg_tmp.append(bg[i])
    n+=1
    if n==Nbg:
      break
  bg_tmp = tf.stack(bg_tmp)

  data = tf.concat([data_tmp,bg_tmp],axis=0)
  data = tf.random.shuffle(data)
  data = tf.transpose(data)
  data_gen = []
  for i,p in zip(range(len(param_list)),param_list):
    data_gen.append(tf.Variable(data[i],name=p,dtype=dtype))
  return data_gen

if __name__=="__main__":
  Ndata = 8065
  Nbg = 3445
  wbg = 0.768331
  data=generate_data(Ndata,Nbg,wbg,1.1,True)
  print(data)
  import pickle
  output = open('toy.pkl','wb')
  pickle.dump(data,output,-1)
  output.close()

