#!/usr/bin/env python3

import json
import math
import os.path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import BFGS, basinhopping, minimize

import iminuit
import tensorflow as tf
from fit_scipy import prepare_data
from tf_pwa.angle import EularAngle, cal_ang_file
from tf_pwa.bounds import Bounds
from tf_pwa.model import FCN, Cache_Model, param_list
from tf_pwa.utils import flatten_np_data, load_config_file, pprint

this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + "/..")




def main(param_name, x, method):
    dtype = "float64"
    w_bkg = 0.768331
    # set_gpu_mem_growth()
    tf.keras.backend.set_floatx(dtype)

    config_list = load_config_file("Resonances")
    data, bg, mcdata = prepare_data(dtype=dtype)
    a = Cache_Model(config_list, w_bkg, data, mcdata, bg=bg, batch=65000)
    try:
        with open("lklpf_params.json") as f:
            param = json.load(f)
            a.set_params(param["value"])
    except:
        pass
    pprint(a.get_params())

    fcn = FCN(a)  # 1356*18

    def LP_minuit(param_name, fixed_var):
        args = {}
        args_name = []
        x0 = []
        bounds_dict = {
            param_name: (fixed_var, fixed_var),
            "Zc_4160_m0:0": (4.1, 4.22),
            "Zc_4160_g0:0": (0, 10),
        }
        for i in a.Amp.trainable_variables:
            args[i.name] = i.numpy()
            x0.append(i.numpy())
            args_name.append(i.name)
            args["error_" + i.name] = 0.1
            if i.name not in bounds_dict:
                bounds_dict[i.name] = (0.0, None)
        for i in bounds_dict:
            if i in args_name:
                args["limit_{}".format(i)] = bounds_dict[i]
        m = iminuit.Minuit(
            fcn,
            forced_parameters=args_name,
            errordef=0.5,
            grad=fcn.grad,
            print_level=2,
            use_array_call=True,
            **args,
        )
        now = time.time()
        with tf.device("/device:GPU:0"):
            print(m.migrad(ncall=10000))  # ,precision=5e-7))
        print(time.time() - now)
        print(m.get_param_states())
        return m

    def LP_sp(param_name, fixed_var):
        args = {}
        args_name = []
        x0 = []
        bnds = []
        bounds_dict = {
            param_name: (fixed_var, fixed_var),
            "Zc_4160_m0:0": (4.1, 4.22),
            "Zc_4160_g0:0": (0, None),
        }
        for i in a.Amp.trainable_variables:
            args[i.name] = i.numpy()
            x0.append(i.numpy())
            args_name.append(i.name)
            if i.name in bounds_dict:
                bnds.append(bounds_dict[i.name])
            else:
                bnds.append((None, None))
            args["error_" + i.name] = 0.1
        now = time.time()
        bd = Bounds(bnds)
        f_g = bd.trans_f_g(fcn.nll_grad)
        callback = lambda x: print(fcn.cached_nll)
        with tf.device("/device:GPU:0"):
            # s = basinhopping(f.nll_grad,np.array(x0),niter=6,disp=True,minimizer_kwargs={"jac":True,"options":{"disp":True}})
            # 优化器
            # s = minimize(fcn.nll_grad,np.array(x0),method="L-BFGS-B",jac=True,bounds=bnds,callback=callback,options={"disp":1,"maxcor":100})
            s = minimize(
                f_g,
                np.array(bd.get_x(x0)),
                method="BFGS",
                jac=True,
                callback=callback,
                options={"disp": 1},
            )
        print("#" * 5, param_name, fixed_var, "#" * 5)
        # print(s)
        return s

    # x=np.arange(0.51,0.52,0.01)
    y = []
    if method == "scipy":
        for v in x:
            y.append(LP_sp(param_name, v).fun)
    elif method == "iminuit":
        for v in x:
            y.append(LP_minuit(param_name, v).get_fmin().fval)
    # print("lklhdx",x)
    # print("lklhdy",y)
    print("\nend\n")
    return y


def lklpf(param_name):
    with open("lklpf_params.json") as f:
        params = json.load(f)
    x_mean = params["value"][param_name]
    x_sigma = params["error"][param_name]
    method = "scipy"  ###
    mode = "bothsides"  # "back&forth"
    if mode == "back&forth":
        x1 = np.arange(
            x_mean - 5 * x_sigma, x_mean + 5 * x_sigma, x_sigma / 2
        )  ###
        x2 = x1[::-1]
        t1 = time.time()
        y1 = main(param_name, x1, method)
        t2 = time.time()
        y2 = main(param_name, x2, method)
        t3 = time.time()
        print(mode, x1, y1, x1, y2[::-1], sep="\n")
    elif mode == "bothsides":
        # x1=np.arange(x_mean,x_mean-5*x_sigma,-x_sigma/2)
        x1 = np.arange(x_mean, x_mean - 100, -10)
        # x2=np.arange(x_mean,x_mean+5*x_sigma,x_sigma/2)
        x2 = np.arange(x_mean, x_mean + 100, 10)
        t1 = time.time()
        y1 = main(param_name, x1, method)
        t2 = time.time()
        y2 = main(param_name, x2, method)
        t3 = time.time()
        print(
            mode,
            list(np.append(x1[::-1], x2)),
            list(np.append(y1[::-1], y2)),
            sep="\n",
        )
    print(param_name, x_mean)
    print("#" * 10, t2 - t1, "#" * 10, t3 - t2)
    """plt.plot(x,yf,"*-",x,yb,"*-")
  plt.title(param_name)
  plt.legend(["forward","backward"])
  plt.savefig("lklhd_prfl")
  plt.clf()"""


if __name__ == "__main__":
    param_list = ["D2_2460_BLS_2_1r:0"]
    for param_name in param_list:
        lklpf(param_name)
    print("\n*** likelihood profile done *****\n")
