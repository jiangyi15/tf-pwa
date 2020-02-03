import tensorflow as tf
import numpy as np
from .utils import is_complex


class Vars(object):
  def __init__(self, model, fix_dic={}, bnd_dic={}):
    self.variables = {}
    self.trainable_vars = []
    self.complex_vars = {}
    self.norm_factor = {} #isopin constr
    self.fix_dic = fix_dic
    self.bnd_dic = bnd_dic
    #self.model = model
    self.add_method = model.add_weight # keras.Model.add_weight 不够通用化


  def add_var(self,name,value=None,range_=None,trainable=True,*arg,**kwarg):
    if name not in self.variables: # a new var
      if name in self.fix_dic: # a fixed var
        value = self.fix_dic[name]
        trainable = False
      if trainable:
        self.trainable_vars.append(name)

      if value is None:
        if range_ is None:
          self.variables[name] = self.add_method(name,trainable=trainable)# 如果没initializer那会是啥？
        else: # random [a,b]
          self.variables[name] = self.add_method(name,initializer=tf.initializers.RandomUniform(*range_),trainable=trainable)
      else: # constant value
        self.variables[name] = self.add_method(name,initializer=tf.initializers.Constant(value),trainable=trainable)
      if name in self.bnd_dic: # set boundary for this var
        self.bnd_dic[name] = Bound(*self.bnd_dic[name],func=None)
    else:
      pass #比如D1_2430r会两次出现
    return self.variables[name]

  def add_complex_var(self,name,polar=True, trainable=True,fix_vals=(1.0,0.0)):
    var_r = name+'r'
    var_i = name+'i'
    if trainable:
      if polar:
        var_r = self.add_var(name=var_r,range_=(0,2.0))
        var_i = self.add_var(name=var_i,range_=(-np.pi,np.pi))
      else:
        var_r = self.add_var(name=var_r,range_=(-1,1))
        var_i = self.add_var(name=var_i,range_=(-1,1))
    else:
      var_r = self.add_var(name=var_r,value=fix_vals[0],False)
      var_i = self.add_var(name=var_i,value=fix_vals[1],False)
    self.complex_vars[name] = [[var_r,var_i],polar] #var_r的值会随着变？

  #def set_complex_pair(self,name,var_r,var_i,polar=True):#add_complex_var相当于先分别add_var再set_complex_pair
  #  if name not in self.complex_vars:
  #    self.complex_vars[name] = [[var_r,var_i],polar]

  def add_norm_factor(self,name,head=None, trainable=True,fix_vals=(1.0,0.0)):
    if not head:
      head = name
    if head not in self.norm_factor:
      if trainable:
        var_r = self.add_var(name=name+'r',range_=(0,2.0))
        var_i = self.add_var(name=name+'i',range_=(-np.pi,np.pi))
      else:
        var_r = self.add_var(name=name+'r',value=fix_vals[0],False)
        var_i = self.add_var(name=name+'i',value=fix_vals[1],False)
      self.norm_factor[head] = [var_r,var_i]
    else:
      if not trainable:
        raise Exception("{0} should be defined before {1}".format(head,name))
      var_i = self.add_var(name=name+'i',range_=(-np.pi,np.pi))
      self.norm_factor[head].append(var_i)


  def get(self,name):
    if name not in self.variables:
      raise Exception("{} not found".format(name))
    return self.variables[name] #tf.Variable
  
  def set(self,name,value):
    if name not in self.variables:
      raise Exception("{} not found".format(name))
    return self.variables[name].assign(value)


  def rp2xy(self,name):
    if name not in self.complex_vars:
      raise Exception("{} not found".format(name))
    if not self.complex_vars[name][1]: # if not polar (already xy)
      return
    r,p = self.complex_vars[name][0]
    x = r * np.cos(p)
    y = r * np.sin(p)
    self.complex_vars[name][0] = [x,y]
    self.complex_vars[name][1] = False

  def xy2rp(self,name):
    if name not in self.complex_vars:
      raise Exception("{} not found".format(name))
    if self.complex_vars[name][1]:
      return
    x,y = self.complex_vars[name][0]
    r = np.sqrt(x*x+y*y)
    p = np.arctan2(y,x)
    self.complex_vars[name][0] = [r,p]
    self.complex_vars[name][1] = True


  def std_polar(self,name):
    if name not in self.complex_vars:
      raise Exception("{} not found".format(name))
    if not self.complex_vars[name][1]:
      self.xy2rp(name)
      self.complex_vars[name][1] = True
    r,p = self.complex_vars[name][0]
    if r<0:
      r.assign(tf.abs(r))
      p.assign_add(np.pi)
    self.std_polar_angle(p)

  @staticmethod
  def std_polar_angle(p,a=-np.pi,b=np.pi)
    twopi = b-a
    while p<=a:
      p.assign_add(twopi)
    while p>=b:
      p.assign_add(-twopi)


  def initialize(self):
    pass

  def get_all(self,after_trans=False): # array (for dict, use self.variables)
    vals = []
    if after_trans:
      for name in self.trainable_vars:
        yval = self.get(name).numpy()
        if name in self.bnd_dic:
          xval = self.bnd_dic[name].get_y2x(yval)
        else:
          xval = yval
        vals.append(xval)
    else:
      for name in self.trainable_vars:
        yval = self.get(name).numpy()
        vals.append(yval)
    return vals
      
  def set_all(self,vals):
    if type(vals)==dict:
      for name in vals:
        self.set(name,vals[name])
    else:
      i = 0
      for name in self.trainable_vars:
        self.set(name,vals[i])
        i+=1

  def rp2xy_all(self,name_list=None):
    if not name_list:
      name_list = self.complex_vars
    for name in name_list:
      self.rp2xy(name)

  def xy2rp_all(self,name_list=None):
    if not name_list:
      name_list = self.complex_vars
    for name in name_list:
      self.xy2rp(name)

  def std_polar_all(self):
    for name in self.complex_vars:
      self.std_polar(name)
    for head in self.norm_factor:
      r = self.norm_factor[head][0]
      if r<0:
        r.assign(tf.abs(r))
        for p in self.norm_factor[head][1:]:
          p.assign_add(np.pi)
      for p in self.norm_factor[head][1:]:
        self.std_polar_angle(p)


  def trans_fcn(self,fcn,grad): # bound transform fcn and grad
    def fcn_t(xvals):
      yvals = xvals
      dydxs = []
      i = 0
      for name in self.trainable_vars:
        if name in self.bnd_dic:
          yvals[i] = self.bnd_dic[name].get_x2y(xvals[i])
          dydxs.append(self.bnd_dic[name].get_dydx(xvals[i]))
        else:
          dydxs.append(1)
        i+=1
      grad_yv = np.array(grad(yvals))
      return fcn(yvals), grad_yv*dydxs
    return fcn_t



import sympy as sy
class Bound(object):
  def __init__(self,a,b,func=None):
    self.lower = a
    self.upper = b
    if func:
      self.func = func # from R (x) to a limited range (y) #Note: y is gls but x is the var in fitting
    else:
      if a==None:
        if b==None:
          self.func = "x"
        else:
          self.func = "b+1-sqrt(x**2+1)"
      else:
        if b==None:
          self.func = "a-1+sqrt(x**2+1)"
        else:
          self.func = "(b-a)*(sin(x)+1)/2+a"
    self.f,self.df,self.inv = self.get_func(self.lower,self.upper)

  def get_func(self,lower,upper):
    x,a,b,y = sy.symbols("x a b y")
    f = sy.sympify(self.func)
    f = f.subs({a:lower,b:upper})
    df = sy.diff(f,x)
    inv = sy.solve(f-y,x)
    if hasattr(inv,"__len__"):
      inv = inv[-1]
    return f,df,inv

  def get_x2y(self,val): # var->gls
    x = sy.symbols('x')
    return self.f.evalf(subs={x:val})
  def get_y2x(self,val): # gls->var
    y = sy.symbols('y')
    return self.inv.evalf(subs={y:val})
  def get_dydx(self,val): # gradient in fitting: dNLL/dx = dNLL/dy * dy/dx
    x = sy.symbols('x')
    return self.df.evalf(subs={x:val})

  def trans_fcn(self,fcn,grad):
    def fcn_t(xs):
      ys = [self.get_x2y(x) for x in xs]


class Variable(Vars): # fitting parameters for the amplitude model
  def __init__(self,model,res,res_decay,polar,**kwarg):
    super(Variable,self).__init__(model,**kwarg)
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
      i = self.add_var(name=head+"i",range_=(-np.pi,np.pi))
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
          tmp_i = self.add_var(name=name+"i",range_=(-np.pi,np.pi))
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


    
