import tensorflow as tf

def fix_value(x):
  def f(shape=None,dtype=None):
    if dtype is not None:
      return tf.Variable(x,dtype=dtype)
    return x
  return f

def rand_value(x):
  def f(shape=None,dtype=None):
    if dtype is not None:
      return tf.Variable(x*tf.random.uniform((),dtype=dtype),dtype=dtype)
    return x*tf.random.uniform((),dtype=dtype)
  return f

def range_value(a,b):
  def f(shape=None,dtype=None):
    return (b-a)*tf.random.uniform((),dtype=dtype)+a
  return f

class Vars(object):
  def __init__(self, env):
    self.env = env
    self.variables = {}

  def add(self,name,size=None,var=None,range=None,trainable=True,**args):
    if name not in self.variables:
      if var is None:
        if size is None:
          self.variables[name] = self.env.add_weight(name,trainable=trainable,**args)
        else:
          if range is None:
            self.variables[name] = self.env.add_weight(name,initializer=rand_value(size),trainable=trainable,**args)
          else:
            self.variables[name] = self.env.add_weight(name,initializer=range_value(*range),trainable=trainable,**args)
      else:
        self.variables[name] = self.env.add_weight(name,initializer=fix_value(var),trainable=trainable,**args)
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
  def __init__(self,name,var,err=0.1,range=(None,None)):
    self.name = name
    self.var = var
    self.err = err
    self.range_ = range
  
  def set_range(self,range_):
    self.range_ = range_
  
  def get_range(self):
    return self.range_
  
  def get_var(self):
    return self.var


    
