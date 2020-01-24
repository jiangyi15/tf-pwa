import os
import tensorflow as tf

# default configurations
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# pylint: disable=no-member
tf_version = int(tf.__version__.split(".")[0])
if tf_version < 2:
  tf.compat.v1.enable_eager_execution()

def set_gpu_mem_growth():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)

if "TF_PWA_GPU_FULL_MEM" in os.environ:
  if os.environ["TF_PWA_GPU_FULL_MEM"] == "0":
    set_gpu_mem_growth()
else:
  set_gpu_mem_growth()
