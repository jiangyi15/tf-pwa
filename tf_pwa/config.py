from .amplitude import AllAmplitude 
from contextlib import contextmanager
from functools import wraps

_config = {
  "amp": AllAmplitude,
  "multi_gpus": False
}

def set_config(name,var):
  """
  set a configuration.
  """
  global _config
  if name in _config:
    _config[name] = var
  else:
    raise Exception("No configuration named {} found.".format(name))

def get_config(name):
  """
  get a configuration.
  """
  if name in _config:
    return _config[name]
  else:
    raise Exception("No configuration named {} found.".format(name))

def regist_config(name,var=None):
  """
  regist a configuration. 
  """
  global _config
  if name in _config:
    raise Exception("Configuration named {} already exists.".format(name))
  if var is None:
    def regist(f):
      _config[name] = f
      return f
    return regist
  else:
    _config[name] = var
    return var

@contextmanager
def temporary_config(name,var):
  tmp = get_config(name)
  set_config(name,var)
  yield var
  set_config(name,tmp)

@contextmanager
def using_amplitude(a):
  tmp = get_config("amp")
  set_config("amp",a)
  yield a
  set_config("amp",tmp)

