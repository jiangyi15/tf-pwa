"""
 Partial Wave Analysis program using Tensorflow

"""
import os
import tensorflow as tf
from .version import __version__
from .utils import set_gpu_mem_growth
#import json

#file_path = os.getcwd()
#config = {}
#if os.path.exists(file_path + "/.config.json"):
  #with open(file_path + "/.config.json") as f:
    #config = json.load(f)

# default configurations
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

tf_version = int(tf.__version__.split(".")[0])
if tf_version < 2:
  tf.compat.v1.enable_eager_execution()

if "TF_PWA_GPU_FULL_MEM" in os.environ :
  if os.environ["TF_PWA_GPU_FULL_MEM"] == "0":
    set_gpu_mem_growth()
else:
  set_gpu_mem_growth()


