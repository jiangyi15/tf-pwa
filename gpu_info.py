import subprocess

support_info = [
  "memory.total",
  "memory.free",
  "memory.used",
  "name",
  "utilization.gpu",
  "utilization.memory"
]

def get_gpu_info(s):
  if s in support_info:
    cmd = "nvidia-smi --query-gpu={} --format=csv,noheader".format(s)
    ret = subprocess.getoutput(cmd)
    return ret.split("\n")
  else:
    raise "Not support"

def get_gpu_total_memory(i=0):
  mem = get_gpu_info("memory.total")
  ret = mem[i].split(" ")[0]
  return float(ret)

def get_gpu_free_memory(i=0):
  mem = get_gpu_info("memory.free")
  ret = mem[i].split(" ")[0]
  return float(ret)

def get_gpu_used_memory(i=0):
  mem = get_gpu_info("memory.used")
  ret = mem[i].split(" ")[0]
  return float(ret)
