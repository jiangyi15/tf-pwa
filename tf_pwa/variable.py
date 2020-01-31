import tensorflow as tf
from numpy import pi
from .utils import is_complex

def fix_value(x): # a dtype constant
  def f(shape=None,dtype=None):
    if dtype is not None:
      return tf.Variable(x,dtype=dtype)
    return x
  return f

def rand_value(x): # random uniform [0,1]
  def f(shape=None,dtype=None):
    if dtype is not None:
      return tf.Variable(x*tf.random.uniform((),dtype=dtype),dtype=dtype)
    return x*tf.random.uniform((),dtype=dtype)
  return f

def range_value(a,b): # random uniform [a,b]
  def f(shape=None,dtype=None):
    return (b-a)*tf.random.uniform((),dtype=dtype)+a
  return f


class Vars(object):
  def __init__(self, add_method):
    self.add_var = add_method # 不过self.env.add_weight不通用化
    self.variables = {}

  def add(self,name,size=None,var=None,range=None,trainable=True,**args):
    if name not in self.variables:
      if var is None:
        if size is None: # random [0,1]
          self.variables[name] = self.add_var(name,trainable=trainable,**args)
        else:
          if range is None: # random [0,size]
            self.variables[name] = self.add_var(name,initializer=rand_value(size),trainable=trainable,**args)
          else: # random [a,b]
            self.variables[name] = self.add_var(name,initializer=range_value(*range),trainable=trainable,**args)
      else: # fix value
        self.variables[name] = self.add_var(name,initializer=fix_value(var),trainable=trainable,**args)
    return self.variables[name]
  
  __call__ = add

  def get(self,name):
    if name not in self.variables:
      raise Exception("{} not found".format(name))
    return self.variables[name]
  
  def set(self,name,var):
    if name not in self.variables:
      raise Exception("{} not found".format(name))
    return self.variables[name].assign(var)


class Variable(object):
  def __init__(self,add_method,res,res_decay,polar,name=None,var=None,err=None,range=(None,None)):
    self.name = name
    self.var = var
    self.err = err
    self.range_ = range

    self.add_var = Vars(add_method) # 通过Vars类来操作variables
    self.res = res
    self.polar = polar # r*e^{ip} or x+iy
    self.res_decay = res_decay
    self.coef = {}
    self.coef_norm = {}

  def params_mass_width(self): # add "mass" "width" fitting parameters
    for i in self.res:
      if "float" in self.res[i]: # variable
        is_float = self.res[i]["float"]
        if is_float:
          self.res[i]["m0"] = self.add_var(name=i+"_m0",var = self.res[i]["m0"],trainable=True)
          self.res[i]["g0"] = self.add_var(name=i+"_g0",var = self.res[i]["g0"],trainable=True)


  def init_res_param(self):
    const_first = True # 第一个共振态系数为1，除非Resonance.yml里指定了某个"total"
    for i in self.res:
      if "total" in self.res[i]:
        const_first = False
    res_tmp = [i for i in self.res]
    res_all = [] # ensure D2_2460 in front of D2_2460p
    # order for coef_head
    while len(res_tmp) > 0:
      i = res_tmp.pop()
      if "coef_head" in self.res[i]: # e.g. "D2_2460" for D2_2460p
        coef_head = self.res[i]["coef_head"]
        if coef_head in res_tmp:
          res_all.append(coef_head)
          res_tmp.remove(coef_head)
      res_all.append(i)
    for i in res_all:
      const_first = self.init_res_param_sig(i,self.res[i],const_first=const_first)
    
  def init_res_param_sig(self,head,config,const_first=False): #head名字，config参数
    self.coef[head] = []
    chain = config["Chain"]
    coef_head = head
    if "coef_head" in config:
      coef_head = config["coef_head"] #这一步把D2_2460p参数变成D2_2460的了
    if "total" in config:
      N_tot = config["total"]
      if is_complex(N_tot):
        N_tot = complex(N_tot)
        rho,phi = N_tot.real,N_tot.imag
      else:
        rho,phi = N_tot #其他类型的complex. raise error?
      r = self.add_var(name=coef_head+"r",var=rho,trainable=False)
      i = self.add_var(name=head+"i",var=phi,trainable=False)
    elif const_first:#先判断有么有total，否则就用const_first
      r = self.add_var(name=coef_head+"r",var=1.0,trainable=False)
      i = self.add_var(name=head+"i",var=0.0,trainable=False)
    else:
      r = self.add_var(name=coef_head+"r",size=2.0)
      i = self.add_var(name=head+"i",range=(-pi,pi))
    self.coef_norm[head] = [r,i]
    if "const" in config: # H里哪一个参数设为常数1
      const = list(config["const"])
    else:
      const = [0,0]
    ls,arg = self.gen_coef(head,0,coef_head+"_",const[0])
    self.coef[head].append(arg)
    ls,arg = self.gen_coef(head,1,coef_head+"_d_",const[1])
    self.coef[head].append(arg)
    return False # const_first
    
  def gen_coef(self,idx,layer,coef_head,const = 0) :
    if const is None:
      const = 0 # set the first to be constant 1 by default
    if isinstance(const,int):
      const = [const] # int2list, in case more than one constant
    ls = self.res_decay[idx][layer].get_ls_list() # allowed l-s pairs
    n_ls = len(ls)
    const_list = []
    for i in const:
      if i<0:
        const_list.append(n_ls + i) # then -1 means the last one
      else:
        const_list.append(i)
    arg_list = []
    for i in range(n_ls):
      l,s = ls[i]
      name = "{head}BLS_{l}_{s}".format(head=coef_head,l=l,s=s)
      if i in const_list:
        tmp_r = self.add_var(name=name+"r",var=1.0,trainable=False)
        tmp_i = self.add_var(name=name+"i",var=0.0,trainable=False)
        arg_list.append((name+"r",name+"i"))
      else :
        if self.polar:
          tmp_r = self.add_var(name=name+"r",size=2.0)
          tmp_i = self.add_var(name=name+"i",range=(-pi,pi))
        else:
          tmp_r = self.add_var(name=name+"r",range=(-1,1))
          tmp_i = self.add_var(name=name+"i",range=(-1,1))
        arg_list.append((name+"r",name+"i"))
    return ls,arg_list


  def get(self,name):
    return self.add_var.get(name)

  def set(self,name,var):
    return self.add_var.set(name,var)

  
  def set_range(self,range_):
    self.range_ = range_
  
  def get_range(self):
    return self.range_
  
  def get_var(self):
    return self.var


    
