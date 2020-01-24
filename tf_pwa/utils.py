import json
import math

class AttrDict(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__


has_yaml = True
try:
  import yaml
except ImportError:
  has_yaml = False

def load_json_file(name):
  with open(name) as f:
    return json.load(f)

def load_yaml_file(name):
  with open(name) as f:
    return yaml.load(f,Loader=yaml.FullLoader)

def load_config_file(name):
  """
  load config file such as Resonances.yml
  """
  if name.endswith("json"):
    return load_json_file(name)
  if has_yaml:
    if name.endswith("yml"):
      return load_yaml_file(name)
    return load_yaml_file(name+".yml")
  else:
    print("no yaml support, using json file")
    return load_json_file(name + ".json")

def flatten_dict_data(data, fun="{}/{}".format):
  if isinstance(data,dict):
    ret = {}
    for i in data:
      tmp = flatten_dict_data(data[i])
      if isinstance(tmp,dict):
        for j in tmp:
          ret[fun(i, j)] = tmp[j]
      else:
        ret[i] = tmp
    return ret
  else :
    return data

flatten_np_data = lambda data: flatten_dict_data(data, fun=lambda x, y: "{}{}".format(y,x[3:]))
  
def error_print(x,err=None):
  if err is None:
    return ("{}").format(x)
  if err <= 0 or math.isnan(err):
    return ("{} ? {}").format(x,err)
  d = math.ceil(math.log10(err))
  b = 10**d
  b_err = err/b
  b_val = x/b
  if b_err < 0.355: #0.100 ~ 0.354
    dig = 2
  elif b_err < 0.950: #0.355 ~ 0.949
    dig = 1
  else: # 0.950 ~ 0.999
    dig = 0
  err = round(b_err,dig) * b
  x = round(b_val,dig)*b
  d_p = dig - d
  if d_p > 0:
    return ("{0:.%df} +/- {1:.%df}"%(d_p,d_p)).format(x,err)
  return ("{0:.0f} +/- {1:.0f}").format(x,err)

def pprint(dicts):
  try:
    s = json.dumps(dicts,indent=2)
    print(s,flush=True)
  except:
    print(dicts,flush=True)

def std_polar(rho,phi):
  if rho<0:
    rho = -rho
    phi+=math.pi
  while phi<-math.pi:
    phi+=2*math.pi
  while phi>math.pi:
    phi-=2*math.pi
  return rho,phi
