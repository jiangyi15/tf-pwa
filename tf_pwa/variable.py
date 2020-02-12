import tensorflow as tf
import numpy as np
import warnings
from .config import regist_config, get_config


class VarsManager(object):
  def __init__(self, dtype=tf.float64):
    self.dtype = dtype
    self.variables = {} # {name:tf.Variable,...}
    self.trainable_vars = [] # [name,...]

    self.complex_vars = {} # {name:polar(bool),...}
    #self.share_r = [] # [[name1,name2],...]
    self.same_list = [] # [[name1,name2],...]

    self.fix_dic = {} # {name:value,...}
    self.bnd_dic = {} # {name:(a,b),...}

    #self.variables_head = {} # {head:[name1,name2],...}

  def add_real_var(self,name, value=None,range_=None,trainable=True):
    if name in self.variables: # not a new var
      if name in self.trainable_vars:
        self.trainable_vars.remove(name)
      warnings.warn("Overwrite variable {}!".format(name))

    if value is None:
      if range_ is None: # random [0,1]
        self.variables[name] = tf.Variable(tf.random.uniform(shape=[],minval=0.,maxval=1.,dtype=self.dtype),name=name,trainable=trainable)
      else: # random [a,b]
        self.variables[name] = tf.Variable(tf.random.uniform(shape=[],minval=range_[0],maxval=range_[1],dtype=self.dtype),name=name,trainable=trainable)
    else: # constant value
      self.variables[name] = tf.Variable(value,name=name,dtype=self.dtype,trainable=trainable)

    if trainable:
      self.trainable_vars.append(name)


  def add_complex_var(self,name, polar=True,trainable=True,fix_vals=(1.0,0.0)):
    var_r = name+'r'
    var_i = name+'i'
    if trainable:
      if polar:
        self.add_real_var(name=var_r,range_=(0,2.0))
        self.add_real_var(name=var_i,range_=(-np.pi,np.pi))
      else:
        self.add_real_var(name=var_r,range_=(-1,1))
        self.add_real_var(name=var_i,range_=(-1,1))
    else:
      self.add_real_var(name=var_r,value=fix_vals[0],trainable=False)
      self.add_real_var(name=var_i,value=fix_vals[1],trainable=False)
    self.complex_vars[name] = polar


  def set_fix(self,name,value): # fix a var (make it untrainable)
    var = tf.Variable(value,name=name,dtype=self.dtype,trainable=False)
    self.variables[name] = var
    try:
      self.trainable_vars.remove(name)
    except:
      warnings.warn("{} has been fixed already!".format(name))

  def set_bound(self,bound_dic,func=None): # set boundary for a var
    for name in bound_dic:
      if name in self.bnd_dic:
        warnings.warn("Overwrite bound of {}!".format(name))
      self.bnd_dic[name] = Bound(*bound_dic[name],func=func)

  def set_share_r(self,name_list): # name_list==[name1,name2,...]
    name_r_list = [name+'r' for name in name_list]
    self.set_same(name_r_list)

  def set_same(self,name_list):
    tmp_list = []
    for name in name_list:
      try:
        del self.complex_vars[name.rstrip('ri')]
      except:
        pass
      for add_list in self.same_list:
        if name in add_list:
          tmp_list += add_list
          self.same_list.remove(add_list)
          break
    name_list += tmp_list
    name_list = list(set(name_list)) #去掉重复元素

    var = self.variables[name_list[0]]
    for name in name_list[1:]:
      if name in self.trainable_vars:
        self.trainable_vars.remove(name)
      else:
        var = self.variables[name] # if one is untrainable, the others will all be untrainable
        if name_list[0] in self.trainable_vars:
          self.trainable_vars.remove(name_list[0])
    for name in name_list:
      self.variables[name] = var
    self.same_list.append(name_list)


  def get(self,name):
    if name not in self.variables:
      raise Exception("{} not found".format(name))
    return self.variables[name] #tf.Variable

  def set(self,name,value):
    if name not in self.variables:
      raise Exception("{} not found".format(name))
    self.variables[name].assign(value)


  def rp2xy(self,name):
    if not self.complex_vars[name]: # if not polar (already xy)
      return
    r = self.variables[name+'r']
    p = self.variables[name+'i']
    x = r * np.cos(p)
    y = r * np.sin(p)
    self.variables[name+'r'] = x
    self.variables[name+'i'] = y
    self.complex_vars[name] = False

  def xy2rp(self,name):
    if self.complex_vars[name]: # if already polar
      return
    x = self.variables[name+'r']
    y = self.variables[name+'i']
    r = np.sqrt(x*x+y*y)
    p = np.arctan2(y,x)
    self.variables[name+'r'] = r
    self.variables[name+'i'] = p
    self.complex_vars[name] = True

  @staticmethod
  def std_polar_angle(p,a=-np.pi,b=np.pi):
    twopi = b-a
    while p<=a:
      p.assign_add(twopi)
    while p>=b:
      p.assign_add(-twopi)
  def std_polar(self,name):
    if not self.complex_vars[name]:
      self.xy2rp(name)
      self.complex_vars[name] = True
    r = self.variables[name+'r']
    p = self.variables[name+'i']
    if r<0:
      r.assign(tf.abs(r))
      p.assign_add(np.pi)
    self.std_polar_angle(p)


  @property
  def trainable_variables(self):
    vars_list = []
    for name in self.trainable_vars:
      vars_list.append(self.variables[name])
    return vars_list

  def get_all(self,after_trans=False): # if bound transf var
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
        yval = self.get(name)
        vals.append(yval)
    return vals # list (for list of tf.Variable use self.get_trainable_vars(); for dict of all vars, use self.variables)

  def set_all(self,vals): # use either dict or list
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

  def std_polar_all(self): # std polar expression: r>0, -pi<p<pi
    for name in self.complex_vars:
      self.std_polar(name)
    '''r_list = set()
    i_list = set()
    for name_list in self.same_list:
      if name_list[0][-1]=='r':
        r_list.add(name_list[0][:-1])
      if name_list[0][-1]=='i':
        i_list.add(name_list[0][:-1])
    ri_list = r_list & i_list
    r_list = r_list - ri_list
    #i_list = i_list - ri_list
    r = self.variables[ri_list[0]+'r']
    p = self.variables[ri_list[0]+'i']
    if r<0:
      r.assign(tf.abs(r))
      p.assign_add(np.pi)
    self.std_polar_angle(p)
    r = self.variables[r_list[0]+'r']
    if r<0:
      r.assign(tf.abs(r))
      for name in r_list:
        p = self.variables[name+'i']
        p.assign_add(np.pi)
        self.std_polar_angle(p)
    else:
      for name in r_list:
        p = self.variables[name+'i']
        self.std_polar_angle(p)'''


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

  def get_func(self,lower,upper): # init func string into sympy f(x) or f(y)
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


def shape_func(f,shape,name,**kwargs):
  if not shape:
    f(name,**kwargs)
  else:
    for i in range(shape[0]):
      shape_func(f,shape[1:],name+'_'+str(i),**kwargs)


regist_config("vm", VarsManager(dtype="float64"))
# vm = VarsManager(dtype=tf.float64)
class Variable(object):
  def __init__(self,name,shape=[],cplx=False,vm=None, **kwargs):
    self.vm = vm
    if vm is None:
      self.vm = get_config("vm")
    self.name = name
    if type(shape)==int:
      shape = [shape]
    self.shape = shape
    self.cplx = cplx
    self.variables = []
    if cplx:
      self.cplx_var(**kwargs)
    else:
      self.real_var(**kwargs)


  def real_var(self, value=None,range_=None,fix=False):
    trainable = not fix
    def func(name,**kwargs):
      self.vm.add_real_var(name, value,range_,trainable)
      self.variables.append(name)
    shape_func(func,self.shape,self.name, value=value,range_=range_,trainable=trainable)

  def cplx_var(self, polar=True,fix=False,fix_which=0,fix_vals=(1.0,0.0)):
    #fix_which = fix_which % self.shape[-1]
    def func(name,**kwargs):
      if self.shape:
        trainable = not (name[-2:]=='_'+str(fix_which))
      else:
        trainable = not fix
      self.vm.add_complex_var(name, polar,trainable,fix_vals)
      self.variables.append(name+'r')
      self.variables.append(name+'i')
    shape_func(func,self.shape,self.name, polar=polar,fix_which=fix_which,fix_vals=fix_vals)


  @property
  def value(self):
    return tf.Variable(self()).numpy()

  def fixed(self,value):
    if not self.shape:
      self.vm.set_fix(self.name,value)
    else:
      raise Exception("Only shape==() real var supports 'fixed' method.")
    

  def r_shareto(self,Var):
    if self.shape != Var.shape:
      raise Exception("Shapes are not the same.")
    if not (self.cplx and Var.cplx):
      raise Exception("Type is not complex var.")
    def func(name,**kwargs):
      self.vm.set_same([self.name+name+'r',Var.name+name+'r'])
    shape_func(func,self.shape,'')

  def sameas(self,Var):
    if self.shape != Var.shape:
      raise Exception("Shapes are not the same.")
    if self.cplx != Var.cplx:
      raise Exception("Types are not the same.")
    def func(name,**kwargs):
      if self.cplx:
        self.vm.set_same([self.name+name+'r',Var.name+name+'r'])
        self.vm.set_same([self.name+name+'i',Var.name+name+'i'])
      else:
        self.vm.set_same([self.name+name,Var.name+name])
    shape_func(func,self.shape,'')


  def __call__(self):
    var_list = np.ones(shape=self.shape).tolist()
    if self.shape:
      def func(name,**kwargs):
        tmp = var_list
        idx_str = name.split('_')[-len(self.shape):]
        for i in idx_str[:-1]:
          tmp = tmp[int(i)]
        if self.cplx:
          if (name in self.vm.complex_vars) and self.vm.complex_vars[name]:
            real = self.vm.variables[name+'r']*tf.cos(self.vm.variables[name+'i'])
            imag = self.vm.variables[name+'r']*tf.sin(self.vm.variables[name+'i'])
            tmp[int(idx_str[-1])] = tf.complex(real,imag)
          else:
            tmp[int(idx_str[-1])] = tf.complex(self.vm.variables[name+'r'],self.vm.variables[name+'i'])
        else:
          tmp[int(idx_str[-1])] = self.vm.variables[name]
      shape_func(func,self.shape,self.name)
    else:
      if self.cplx:
        if (self.name in self.vm.complex_vars) and self.vm.complex_vars[self.name]:
          real = self.vm.variables[self.name+'r']*tf.cos(self.vm.variables[self.name+'i'])
          imag = self.vm.variables[self.name+'r']*tf.sin(self.vm.variables[self.name+'i'])
          var_list = tf.complex(real,imag)
        else:
          var_list = tf.complex(self.vm.variables[self.name+'r'],self.vm.variables[self.name+'i'])
      else:
        var_list = self.vm.variables[self.name]

    #return tf.stack(var_list)
    return var_list


if __name__ == "__main__":
  m = Variable("R_m",value=2.1) #trainable is True by default
  g_ls = Variable("A2BR_H",shape=[3],cplx=True)
  fcr = Variable("R_total",cplx=True)
  m1 = Variable("R1_m",value=2.3)
  g_ls1 = Variable("A2BR1_H",shape=[3],cplx=True)
  fcr1 = Variable("R1_total",cplx=True)

  print(g_ls.value)
  print(g_ls())
  
  m.fixed(2.4)
  g_ls.sameas(g_ls1)
  fcr.r_shareto(fcr1)



'''class AmpVar(object): # fitting parameters for the amplitude model
  def __init__(self,res,res_decay,polar,**kwarg):
    #super(AmpVar,self).__init__(**kwarg)
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
          self.res[i]["m0"] = Variable(name=i+"_m",value = self.res[i]["m0"]) #然后self.res[i]["m0"]就成一个变量了（BW里会调用）
        if "g" in floating:
          self.res[i]["g0"] = Variable(name=i+"_g",value = self.res[i]["g0"])

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
      #fcr = Variable(head,cplx=True,fix_which=True,fix_vals=(rho,phi))
      r = Variable(coef_head+"r",value=rho,fix=True)
      i = Variable(head+"i",value=phi,fix=True)
    elif const_first:#先判断有么有total，否则就用const_first
      #fcr = Variable(head,cplx=True,fix_which=True)
      r = Variable(coef_head+"r",value=1.0,fix=True)
      i = Variable(head+"i",value=0.0,fix=True)
    else:
      #fcr = Variable(head,cplx=True)
      r = Variable(coef_head+"r",range_=(0,2.0))
      i = Variable(head+"i",range_=(-np.pi,np.pi))
    self.coef_norm[head] = [r,i] #fcr
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
        Variable(name,cplx=True,fix_which=True)
        #tmp_r = self.add_real_var(name=name+"r",value=1.0,trainable=False)
        #tmp_i = self.add_real_var(name=name+"i",value=0.0,trainable=False)
        arg_list.append((name+"r",name+"i"))
      else :
        if self.polar:
          Variable(name,cplx=True)
          #tmp_r = self.add_real_var(name=name+"r",range_=(0,2.0))
          #tmp_i = self.add_real_var(name=name+"i",range_=(-np.pi,np.pi))
        else:
          Variable(name,cplx=True,polar=False)
          #tmp_r = self.add_real_var(name=name+"r",range_=(-1,1))
          #tmp_i = self.add_real_var(name=name+"i",range_=(-1,1))
        arg_list.append((name+"r",name+"i"))
    return ls,arg_list
'''
