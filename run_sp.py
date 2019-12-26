import tensorflow as tf
import time as t
import os
from fit_scipy import main
#import likelihood_profile

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

### fitting
print("-----Start-----",i)
t1=t.time()
main()
t2=t.time()
print("-----Complete in %.2fs-----"%(t2-t1))

### likelihood profile
#likelihood_profile.main(param_name,x,method)
