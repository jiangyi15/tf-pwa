import json
from model import *

class fcn(object):
  def __init__(self,model,data,mcdata,batch=5000):
    self.model = model
    self.data = data
    self.mcdata = mcdata
    self.batch = batch
  def __call__(self,x):
    train_vars = self.model.Amp.trainable_variables
    n_var = len(train_vars)
    for i in range(n_var):
      train_vars[i].assign(x[i])
    nll,g = self.model.nll_gradient(self.data,self.mcdata,batch=self.batch)
    return nll.numpy()
  
  def grad(self,x):
    now = time.time()
    train_vars = self.model.Amp.trainable_variables
    n_var = len(train_vars)
    for i in range(n_var):
      train_vars[i].assign(x[i])
    nll,g = self.model.nll_gradient(self.data,self.mcdata,batch=self.batch)
    self.grads = [ i.numpy() for i in g]
    print("nll:", nll," time :",time.time() - now)
    return np.array(self.grads)

def main():
  dtype = "float32"
  set_gpu_mem_growth()
  tf.keras.backend.set_floatx(dtype)
  with open("Resonances.json") as f:  
    config_list = json.load(f)
  a = Model(config_list,0.768331)
  #print(a.Amp.coef)
  with open("need.json") as f:  
    param = json.load(f)
    a.set_params(param)
  s = json.dumps(a.get_params(),indent=2)
  print(s)
  def load_data(fname):
    dat = []
    with open(fname) as f:
      tmp = json.load(f)
      for i in param_list:
        tmp_data = tf.Variable(tmp[i],name=i,dtype=dtype)
        dat.append(tmp_data)
    return dat
  data = load_data("./data/toymc.json")
  #bg = load_data("./data/bg_ang_n4.json")
  mcdata = load_data("./data/toyPHSP.json")
  
  t = time.time()
  nll,g = a.nll_gradient(data,mcdata,batch=50000)
  print("Time:",time.time()-t)
  print(nll)
  print(g)
  if False: #check gradient
    for i in a.Amp.trainable_variables:
      tmp = i.numpy()
      i.assign(tmp+1e-3)
      nll_0,_ = a.nll_gradient(data,mcdata,batch=50000)
      i.assign(tmp-1e-3)
      nll_1,_ = a.nll_gradient(data,mcdata,batch=50000)
      i.assign(tmp)
      print(i,(nll_0-nll_1).numpy()/2e-3)
  
  #exit()
  #print(a.get_params())

  import iminuit 
  f = fcn(a,data,mcdata,100000)
  args = {}
  args_name = []
  x0 = []
  for i in a.Amp.trainable_variables:
    args[i.name] = i.numpy()
    x0.append(i.numpy())
    args_name.append(i.name)
    args["error_"+i.name] = 0.1
  ##args["limit_Zc_4160_m0:0"] = (4.1,4.22)
  m = iminuit.Minuit(f,forced_parameters=args_name,errordef = 0.5,grad=f.grad,print_level=2,use_array_call=True,**args)
  now = time.time()
  with tf.device('/device:GPU:0'):
    print(m.migrad(ncall=10000))
  print(time.time() - now)
  print(m.get_param_states())
  with open("need_params.json","w") as f:
    json.dump(a.get_params(),f,indent=2)
  #try :
    #print(m.minos())
  #except RuntimeError as e:
    #print(e)
  #print(m.get_param_states())
  #with tf.device('/device:GPU:0'):
    #print(a.nll(data,bg,mcdata))#.collect_params())
  #print(a.Amp.trainable_variables)
  
if __name__=="__main__":
  main()
