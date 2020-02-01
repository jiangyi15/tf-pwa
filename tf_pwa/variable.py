import tensorflow as tf
from numpy import pi
from .utils import is_complex

'''
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
'''

class Vars(object):
  def __init__(self, add_method, fix_dic={}, bnd_dic={}):
    self.variables = {}
    self.trainable_vars = []
    self.fix_dic = fix_dic
    self.bnd_dic = bnd_dic
    self.add_method = add_method # keras.Model.add_weight 不够通用化


  def add_var(self,name,value=None,range_=None,trainable=True,*arg,**kwarg):
    if name not in self.variables: # a new var
      if name in self.fix_dic: # a fixed var
        value = self.fix_dic[name]
        trainable = False
      if name in self.bnd_dic: # set boundary for this var
        pass
      if trainable:
        self.trainable_vars.append(name)

      if value is None:
        if range_ is None:
          self.variables[name] = self.add_method(name,trainable=trainable)# 如果没iniitializer那会是啥？
        else: # random [a,b]
          self.variables[name] = self.add_method(name,initializer=tf.initializers.RandomUniform(*range_),trainable=trainable)
      else: # constant value
        self.variables[name] = self.add_method(name,initializer=tf.initializers.Constant(value),trainable=trainable)
    
    else:
      pass #比如D1_2430r会两次出现
    return self.variables[name]


  def get(self,name):
    if name not in self.variables:
      raise Exception("{} not found".format(name))
    return self.variables[name]
  
  def set(self,name,value):
    if name not in self.variables:
      raise Exception("{} not found".format(name))
    return self.variables[name].assign(value)


class Variable(Vars): # fitting parameters for the amplitude model
  def __init__(self,add_method,res,res_decay,polar,**kwarg):
    super(Variable,self).__init__(add_method,**kwarg)
    self.res = res
    self.polar = polar # r*e^{ip} or x+iy
    self.res_decay = res_decay
    self.coef = {}
    self.coef_norm = {}


  def init_fit_params(self):
    self.init_params_mass_width()
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
      const_first = self.init_partial_wave_coef(i,self.res[i],const_first=const_first)

  def init_params_mass_width(self): # add "mass" "width" fitting parameters
    for i in self.res:
      if "float" in self.res[i]: # variable
        floating = self.res[i]["float"]
        floating = str(floating)
        if "m" in floating:
          self.res[i]["m0"] = self.add_var(name=i+"_m",value = self.res[i]["m0"]) #然后self.res[i]["m0"]就成一个变量了（BW里会调用）
        if "g" in floating:
          self.res[i]["g0"] = self.add_var(name=i+"_g",value = self.res[i]["g0"])
    
  def init_partial_wave_coef(self,head,config,const_first=False): #head名字，config参数
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
      r = self.add_var(name=coef_head+"r",value=rho,trainable=False)
      i = self.add_var(name=head+"i",value=phi,trainable=False)
    elif const_first:#先判断有么有total，否则就用const_first
      r = self.add_var(name=coef_head+"r",value=1.0,trainable=False)
      i = self.add_var(name=head+"i",value=0.0,trainable=False)
    else:
      r = self.add_var(name=coef_head+"r",range_=(0,2.0))
      i = self.add_var(name=head+"i",range_=(-pi,pi))
    self.coef_norm[head] = [r,i]
    if "const" in config: # H里哪一个参数设为常数1
      const = list(config["const"])
    else:
      const = [0,0]
    ls,arg = self.gen_coef_gls(head,0,coef_head+"_",const[0])
    self.coef[head].append(arg)
    ls,arg = self.gen_coef_gls(head,1,coef_head+"_d_",const[1])
    self.coef[head].append(arg)
    return False # const_first
    
  def gen_coef_gls(self,idx,layer,coef_head,const = 0) :
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
        tmp_r = self.add_var(name=name+"r",value=1.0,trainable=False)
        tmp_i = self.add_var(name=name+"i",value=0.0,trainable=False)
        arg_list.append((name+"r",name+"i"))
      else :
        if self.polar:
          tmp_r = self.add_var(name=name+"r",range_=(0,2.0))
          tmp_i = self.add_var(name=name+"i",range_=(-pi,pi))
        else:
          tmp_r = self.add_var(name=name+"r",range_=(-1,1))
          tmp_i = self.add_var(name=name+"i",range_=(-1,1))
        arg_list.append((name+"r",name+"i"))
    return ls,arg_list


  def set_range(self,range_):
    self.range_ = range_
  
  def get_range(self):
    return self.range_
  
  def get_var(self):
    return self.var


    
