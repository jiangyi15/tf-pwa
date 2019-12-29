import json

has_yaml = True
try:
  import yaml
except ImportError:
  has_yaml = True

def load_json_file(name):
  with open(name) as f:
    return json.load(f)

def load_yaml_file(name):
  with open(name) as f:
    return yaml.load(f,Loader=yaml.FullLoader)

def load_config_file(name):
  if name.endswith("json"):
    return load_json_file(name)
  if has_yaml:
    if name.endswith("yml"):
      return load_yaml_file(name)
    return load_yaml_file(name+".yml")
  else:
    return load_json_file(name + ".json")
    
  
