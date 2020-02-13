import tensorflow as tf
import numpy as np
import warnings
#from .config import regist_config


'''
vm = VarsManager(dtype=tf.float64)

vm.set_fix(var_name,value)#var_name是实变量的name（复变量name的两个实分量分别叫namer，namei）
vm.set_bound({var_name:(a,b)},func="(b-a)*(sin(x)+1)/2+a")
vm.set_share_r([var_name1,var_name2])#var_name是复变量的name
vm.set_all(init_params)
vm.std_polar_all()
vm.trans_fcn(fcn,grad)#bound转换
'''


class VarsManager(object):
  def __init__(self, dtype=tf.float64):
    self.dtype = dtype
    self.variables = {} # {name:tf.Variable,...}
    self.trainable_vars = [] # [name,...]
    self.complex_vars = {} # {name:polar(bool),...}
    self.same_list = [] # [[name1,name2],...]

    self.fix_dic = {} # {name:value,...}
    self.bnd_dic = {} # {name:(a,b),...}

    self.var_head = {} # {head:[name1,name2],...}

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

  def remove_var(self,name,cplx=False):
    if cplx:
      del self.complex_vars[name]
      name_r = name+'r'
      name_i = name+'i'
      if self.variables[name_r].trainable:
        self.trainable_vars.remove(name_r)
      if self.variables[name_i].trainable:
        self.trainable_vars.remove(name_i)
      for l in self.same_list:
        if name_r in l:
          l.remove(name_r)
        if name_i in l:
          l.remove(name_i)
      del self.variables[name_r]
      del self.variables[name_i]
    else:
      if self.variables[name].trainable:
        self.trainable_vars.remove(name)
      for l in self.same_list:
        if name in l:
          l.remove(name)
      del self.variables[name]


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
    for name in name_r_list:
      self.complex_vars[name] = name_r_list

  def set_same(self,name_list,cplx=False):
    tmp_list = []
    for name in name_list:
      for add_list in self.same_list:
        if name in add_list:
          tmp_list += add_list
          self.same_list.remove(add_list)
          break
    name_list += tmp_list
    name_list = list(set(name_list)) #去掉重复元素
    def same_real(name_list):
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
    
    if cplx:
      same_real([name+'r' for name in name_list])
      same_real([name+'i' for name in name_list])
    else:
      same_real(name_list)
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
    if self.complex_vars[name]!=True: # if not polar (already xy)
      return
    r = self.variables[name+'r']
    p = self.variables[name+'i']
    x = r * np.cos(p)
    y = r * np.sin(p)
    self.variables[name+'r'] = x
    self.variables[name+'i'] = y
    self.complex_vars[name] = False
    for l in self.same_list:
      if name in l:
        for i in l:
          self.complex_vars[i] = False
        break

  def xy2rp(self,name):
    if self.complex_vars[name]!=False: # if already polar
      return
    x = self.variables[name+'r']
    y = self.variables[name+'i']
    r = np.sqrt(x*x+y*y)
    p = np.arctan2(y,x)
    self.variables[name+'r'] = r
    self.variables[name+'i'] = p
    self.complex_vars[name] = True
    for l in self.same_list:
      if name in l:
        for i in l:
          self.complex_vars[i] = True
        break


  @property
  def trainable_variables(self):
    vars_list = []
    for name in self.trainable_vars:
      vars_list.append(self.variables[name])
    return vars_list

  def get_all_val(self,after_trans=False): # if bound transf var
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
    return vals # list (for list of tf.Variable use self.trainable_variables; for dict of all vars, use self.variables)

  def get_all_dic(self,trainable_only=False):
    self.std_polar_all()
    dic = {}
    if trainable_only:
      for i in self.trainable_vars:
        dic[i] = self.variables[i].numpy()
    else:
      for i in self.variables:
        dic[i] = self.variables[i].numpy()
    return dic

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

  @staticmethod
  def std_polar_angle(p,a=-np.pi,b=np.pi):
    twopi = b-a
    while p<=a:
      p.assign_add(twopi)
    while p>=b:
      p.assign_add(-twopi)
  def std_polar(self,name):
    self.xy2rp(name)
    r = self.variables[name+'r']
    p = self.variables[name+'i']
    if r<0:
      r.assign(tf.abs(r))
      p.assign_add(np.pi)
      if type(self.complex_vars[name])==list:
        for name_r in self.complex_vars[name]:
          pp = self.variables[name_r[:-1]+'i']
          pp.assign_add(np.pi)
    self.std_polar_angle(p)

  def std_polar_all(self): # std polar expression: r>0, -pi<p<pi
    for name in self.complex_vars:
      self.std_polar(name)

  def trans_params(self,polar):
    if polar:
      self.std_polar_all()
    else:
      self.rp2xy_all()


  def trans_fcn_grad(self,fcn_grad): # bound transform fcn and grad
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
      fcn, grad_yv = fcn_grad(yvals)
      grad = np.array(grad_yv)*dydxs
      return fcn, grad
    return fcn_t


import sympy as sy
class Bound(object):
  def __init__(self,a,b,func=None):
    if a!=None and b!=None and a>b:
      raise Exception("Lower bound is larger than upper bound!")
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
    return float(self.f.evalf(subs={x:val}))
  def get_y2x(self,val): # gls->var
    y = sy.symbols('y')
    if self.lower!=None and val<self.lower:
      val = self.lower
    elif self.upper!=None and val>self.upper:
      val = self.upper
    return float(self.inv.evalf(subs={y:val}))
  def get_dydx(self,val): # gradient in fitting: dNLL/dx = dNLL/dy * dy/dx
    x = sy.symbols('x')
    return float(self.df.evalf(subs={x:val}))



def shape_func(f,shape,name,**kwargs):
  if not shape:
    f(name,**kwargs)
  else:
    for i in range(shape[0]):
      shape_func(f,shape[1:],name+'_'+str(i),**kwargs)
#regist_config("vm", VarsManager(dtype="float64"))
vm = VarsManager(dtype=tf.float64)
class Variable(object):
  def __init__(self,name,shape=[],cplx=False,vm=vm, **kwargs):
    self.vm = vm
    self.name = name
    if name in vm.var_head:
      warnings.warn("Overwrite Variable {}!".format(name))
      for i in vm.var_head[name]:
        vm.remove_var(i,cplx)
    self.vm.var_head[self.name] = []
    if type(shape)==int:
      shape = [shape]
    self.shape = shape
    self.cplx = cplx
    if cplx:
      self.cplx_var(**kwargs)
    else:
      self.real_var(**kwargs)


  def real_var(self, value=None,range_=None,fix=False):
    trainable = not fix
    def func(name,**kwargs):
      self.vm.add_real_var(name, value,range_,trainable)
      self.vm.var_head[self.name].append(name)
    shape_func(func,self.shape,self.name, value=value,range_=range_,trainable=trainable)

  def cplx_var(self, polar=True,fix=False,fix_which=0,fix_vals=(1.0,0.0)):
    #fix_which = fix_which % self.shape[-1]
    def func(name,**kwargs):
      if self.shape:
        trainable = not (name[-2:]=='_'+str(fix_which))
      else:
        trainable = not fix
      self.vm.add_complex_var(name, polar,trainable,fix_vals)
      self.vm.var_head[self.name].append(name)
    shape_func(func,self.shape,self.name, polar=polar,fix_which=fix_which,fix_vals=fix_vals)


  @property
  def value(self):
    return tf.Variable(self()).numpy()

  @property
  def variables(self):
    return self.vm.var_head[self.name]

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
      self.vm.set_same([self.name+name,Var.name+name],cplx=self.cplx)
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
        name = self.name
        if (name in self.vm.complex_vars) and self.vm.complex_vars[name]:
          real = self.vm.variables[name+'r']*tf.cos(self.vm.variables[name+'i'])
          imag = self.vm.variables[name+'r']*tf.sin(self.vm.variables[name+'i'])
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

