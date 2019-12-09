#!/usr/bin/env python3

import json
from model import *
from angle import cal_ang_file,EularAngle

def train_one_step(model, optimizer):
  nll,grads = model.cal_nll_gradient({})
  optimizer.apply_gradients(zip(grads, model.Amp.trainable_variables))
  return nll,grads

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

def main():
  dtype = "float64"
  set_gpu_mem_growth()
  tf.keras.backend.set_floatx(dtype)
  with open("Resonances.json") as f:  
    config_list = json.load(f)
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
  with tf.device('/device:GPU:0'):
    data = load_data("data")
    bg = load_data("bg")
    mcdata = load_data("PHSP")
    a = Cache_Model(config_list,0.768331,data,mcdata,bg=bg,batch=65000)
  #print(a.Amp.coef)
  
  try :
    with open("finalss_params.json") as f:  
      param = json.load(f)
      a.set_params(param)
  except:
    pass
  s = json.dumps(a.get_params(),indent=2)
  print(s)
  #a.Amp(data)
  #exit()
  data_w,weights = data,1.0#a.get_weight_data(data,bg)
  t = time.time()
  nll,g = a.cal_nll_gradient()#data_w,mcdata,weight=weights,batch=50000)
  print("Time:",time.time()-t)
  print(nll)
  #he = np.array([[j.numpy() for j in i] for i in h])
  #print(he)
  #ihe = np.linalg.inv(he)
  #print(ihe)
  if False: #check gradient
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
  
  import iminuit 
  f = FCN(a)
  args = {}
  args_name = []
  x0 = []
  for i in a.Amp.trainable_variables:
    args[i.name] = i.numpy()
    x0.append(i.numpy())
    args_name.append(i.name)
    args["error_"+i.name] = 0.1
  bounds_dict = {
      "Zc_4160_m0:0":(4.1,4.22),
      "Zc_4160_g0:0":(0,10)
  }
  for i in bounds_dict:
    if i in args_name:
      args["limit_{}".format(i)] = bounds_dict[i]
  m = iminuit.Minuit(f,forced_parameters=args_name,errordef = 0.5,grad=f.grad,print_level=2,use_array_call=True,**args)
  now = time.time()
  with tf.device('/device:GPU:0'):
    print(m.migrad(ncall=10000))#,precision=5e-7))
  print(time.time() - now)
  print(m.get_param_states())
  with open("final_params.json","w") as f:
    json.dump(a.get_params(),f,indent=2)
  #try :
    #print(m.minos())
  #except RuntimeError as e:
    #print(e)
  #print(m.get_param_states())
  #with tf.device('/device:GPU:0'):
    #print(a.nll(data,bg,mcdata))#.collect_params())
  #print(a.Amp.trainable_variables)
  t = time.time()
  a_h = Cache_Model(config_list,0.768331,data,mcdata,bg=bg,batch=26000)
  a_h.set_params(a.get_params())
  nll,g,h = a_h.cal_nll_hessian()#data_w,mcdata,weight=weights,batch=50000)
  print("Time:",time.time()-t)
  print(nll)
  print([i.numpy() for i in g])
  print(h.numpy())
  print(np.linalg.inv(h.numpy()))
  
if __name__=="__main__":
  main()
