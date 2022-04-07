#!/usr/bin/env python3

import json
import math
import os
import os.path
import subprocess
import sys
import threading
import time
from sys import platform

import numpy as np
import pymultinest
import tensorflow as tf
from numpy import cos, pi
from pymultinest.solve import solve
from scipy.optimize import BFGS, basinhopping, minimize

from tf_pwa.amplitude import AllAmplitude, param_list
from tf_pwa.angle import cal_ang_file, cal_ang_file4
from tf_pwa.bounds import Bounds
from tf_pwa.fitfractions import cal_fitfractions
from tf_pwa.model import FCN, Cache_Model, param_list
from tf_pwa.utils import error_print, flatten_np_data, load_config_file, pprint

this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + "/..")


try:
    os.mkdir("chains")
except OSError:
    pass


def show(filepath):
    """open the output (pdf) file for the user"""
    if os.name == "mac" or platform == "darwin":
        subprocess.call(("open", filepath))
    elif os.name == "nt" or platform == "win32":
        os.startfile(filepath)
    elif platform.startswith("linux"):
        subprocess.call(("xdg-open", filepath))


def prepare_data(dtype="float64", model="3"):
    fname = [
        ["./data/data4600_new.dat", "data/Dst0_data4600_new.dat"],
        ["./data/bg4600_new.dat", "data/Dst0_bg4600_new.dat"],
        ["./data/PHSP4600_new.dat", "data/Dst0_PHSP4600_new.dat"],
    ]
    tname = ["data", "bg", "PHSP"]
    data_np = {}
    for i in range(len(tname)):
        if model == "3":
            data_np[tname[i]] = cal_ang_file(fname[i][0], dtype)
        elif model == "4":
            data_np[tname[i]] = cal_ang_file4(fname[i][0], fname[i][1], dtype)

    def load_data(name):
        dat = []
        tmp = flatten_np_data(data_np[name])
        for i in param_list:
            tmp_data = tf.Variable(tmp[i], name=i, dtype=dtype)
            dat.append(tmp_data)
        return dat

    # with tf.device('/device:GPU:0'):
    data = load_data("data")
    bg = load_data("bg")
    mcdata = load_data("PHSP")
    return data, bg, mcdata


def main():
    dtype = "float64"
    w_bkg = 0.768331
    # set_gpu_mem_growth()
    tf.keras.backend.set_floatx(dtype)
    # open Resonances list as dict
    config_list = load_config_file("Resonances")

    data, bg, mcdata = prepare_data(dtype=dtype)  # ,model="4")

    amp = AllAmplitude(config_list)
    a = Cache_Model(
        amp, w_bkg, data, mcdata, bg=bg, batch=65000
    )  # ,constrain={"Zc_4160_g0:0":(0.1,0.1)})
    try:
        with open("final_params.json") as f:
            param = json.load(f)
            # print("using init_params.json")
            if "value" in param:
                a.set_params(param["value"])
            else:
                a.set_params(param)
    except:
        pass
    # print(a.Amp(data))
    # exit()
    # pprint(a.get_params())
    # print(data,bg,mcdata)
    # t = time.time()
    # nll,g = a.cal_nll_gradient()#data_w,mcdata,weight=weights,batch=50000)
    # print("nll:",nll,"Time:",time.time()-t)
    # exit()
    fcn = FCN(a)

    # fit configure
    args = {}
    args_name = []
    x0 = []
    bnds = []
    bounds_dict = {"Zc_4160_m0:0": (4.1, 4.22), "Zc_4160_g0:0": (0, None)}

    for i in a.Amp.trainable_variables:
        args[i.name] = i.numpy()
        x0.append(i.numpy())
        args_name.append(i.name)
        if i.name in bounds_dict:
            bnds.append(bounds_dict[i.name])
        else:
            bnds.append((None, None))
        args["error_" + i.name] = 0.1

    bnds_cube = []
    for a, b in bnds:
        if a is None:
            if b is None:
                bnds_cube.append((200, -100))
            else:
                bnds_cube.append((b + 100, -100))
        else:
            if b is None:
                bnds_cube.append((100 - a, a))
            else:
                bnds_cube.append((b - a, a))
    # bnds_cube=[(0,1),(0,1)]
    cube_size = np.array(bnds_cube)

    def Prior(cube):
        cube[:] = cube * cube_size[:, 0] + cube_size[:, 1]
        return cube

    def LogLikelihood(cube):
        # return -np.sum(cube*cube)
        ret = fcn(cube)
        return -ret

    prefix = "chains/1-"
    with open("%sparams.json" % prefix, "w") as f:
        json.dump(args_name, f, indent=2)
    n_params = len(args_name)

    # we want to see some output while it is running
    # progress = pymultinest.ProgressPlotter(n_params = n_params, outputfiles_basename=prefix);
    # progress.start()
    # threading.Timer(30, show, [prefix+"phys_live.points.pdf"]).start() # delayed opening
    now = time.time()
    solution = solve(
        LogLikelihood,
        Prior,
        n_dims=n_params,
        outputfiles_basename=prefix,
        importance_nested_sampling=False,
        verbose=True,
    )
    print("Time for fitting:", time.time() - now)
    # progress.stop()
    # print(solution)
    print()
    print("evidence: %(logZ).1f +- %(logZerr).1f" % solution)
    print()
    print("parameter values:")
    for name, col in zip(args_name, solution["samples"].transpose()):
        print("%15s : %.3f +- %.3f" % (name, col.mean(), col.std()))
    # use: solution.logZ, solution.logZerr, solution.samples


if __name__ == "__main__":
    main()
