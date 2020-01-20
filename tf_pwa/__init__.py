"""
 Partial Wave Analysis program using Tensorflow

"""
import os
import tensorflow as tf
from .version import __version__
from .utils import set_gpu_mem_growth

# default configurations
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

tf_version = int(tf.__version__.split(".")[0])
if tf_version < 2:
  tf.compat.v1.enable_eager_execution()

set_gpu_mem_growth()


