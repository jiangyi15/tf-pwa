#!/usr/bin/env python3
from tf_pwa.model_new import CachedModel, FCN
import tensorflow as tf
import time
import numpy as np
import json
import os
from scipy.optimize import minimize, BFGS, basinhopping
import tf_pwa
from tf_pwa.angle import cal_ang_file, cal_ang_file4
from tf_pwa.utils import load_config_file, flatten_np_data, pprint, error_print, std_polar
from tf_pwa.fitfractions import cal_fitfractions, cal_fitfractions_no_grad
import math
from tf_pwa.bounds import Bounds
from plot_amp import calPWratio

from tf_pwa.amp import AmplitudeModel, DecayGroup, HelicityDecay, Particle

from tf_pwa.data import prepare_data_from_decay


def load_cached_data(cached_data_file="cached_data.npy"):
    cached_dir = "./cached_dir"
    if not os.path.exists(cached_dir):
        return None
    cached_path = os.path.join(cached_dir, cached_data_file)
    if os.path.exists(cached_path):
        cached_data = tf_pwa.load_data(cached_path)
        return cached_data
    return None


def save_cached_data(cached_data, cached_data_file="cached_data.npy"):
    cached_dir = "./cached_dir"
    if not os.path.exists(cached_dir):
        os.mkdir(cached_dir)
    cached_path = os.path.join(cached_dir, cached_data_file)
    tf_pwa.save_data(cached_path, cached_data)


def prepare_data(decs, dtype="float64"):
    fname = [
      ["./data/data4600_new.dat", "data/Dst0_data4600_new.dat"],
      ["./data/bg4600_new.dat", "data/Dst0_bg4600_new.dat"],
      ["./data/PHSP4600_new.dat", "data/Dst0_PHSP4600_new.dat"]
    ]
    tname = ["data", "bg", "PHSP"]
    cached_data = load_cached_data()
    if False and cached_data is not None:
        data = cached_data["data"]
        bg = cached_data["bg"]
        mcdata = cached_data["PHSP"]
        print("using cached data")
        return data, bg, mcdata
    data_np = {}
    for i, name in enumerate(fname):
        data_np[tname[i]] = prepare_data_from_decay(name[0], decs)

    data, bg, mcdata = [data_np[i] for i in tname]
    save_cached_data({"data": data, "bg": bg, "PHSP": mcdata})
    return data, bg, mcdata


def cal_hesse_error(Amp, val, w_bkg, data, mcdata, bg, args_name, batch):
    a_h = CachedModel(Amp, w_bkg, data, mcdata, bg=bg, batch=batch)
    a_h.set_params(val)
    t = time.time()
    nll, g, h = a_h.cal_nll_hessian()  # data_w,mcdata,weight=weights,batch=50000)
    print("Time for calculating errors:", time.time() - t)
    # print(nll)
    # print([i.numpy() for i in g])
    # print(h.numpy())
    inv_he = np.linalg.pinv(h.numpy())
    np.save("error_matrix.npy", inv_he)
    # print("edm:",np.dot(np.dot(inv_he,np.array(g)),np.array(g)))
    return inv_he


def fit(method="BFGS", init_params="init_params.json", hesse=True, frac=True):
    POLAR = True  # fit in polar coordinates. should be consistent with init_params.json if any
    GEN_TOY = False  # use toy data (mcdata and bg stay the same). REMEMBER to update gen_params.json

    dtype = "float64"
    w_bkg = 0.768331
    # set_gpu_mem_growth()
    tf.keras.backend.set_floatx(dtype)
    # open Resonances list as dict
    config_list = load_config_file("Resonances")

    a = Particle("A", J=1, P=-1, spins=(-1, 1))
    b = Particle("B", J=1, P=-1)
    c = Particle("C", J=0, P=-1)
    d = Particle("D", J=1, P=-1)
    for i in config_list:
        config = config_list[i]
        res = Particle(i, config["J"], config["Par"], mass=config["m0"], width=config["g0"])
        chain = config["Chain"]
        if chain < 0:
            HelicityDecay(a, [res, c])
            HelicityDecay(res, [b, d])
        elif chain < 100:
            HelicityDecay(a, [res, d])
            HelicityDecay(res, [b, c])
        elif chain < 200:
            HelicityDecay(a, [res, b])
            HelicityDecay(res, [d, c])

    decs = DecayGroup(a.chain_decay())
    data, bg, mcdata = prepare_data(decs, dtype=dtype)

    amp = AmplitudeModel(decs)
    model = CachedModel(amp, data, mcdata, bg=bg, w_bkg=w_bkg)
    now = time.time()
    print(model.nll(data, mcdata))
    print(time.time() - now)
    now = time.time()
    print(model.nll_grad(data, mcdata))
    print(time.time() - now)
    fcn = FCN(model)
    print(fcn.grad({}))



def main():
    import argparse
    parser = argparse.ArgumentParser(description="simple fit scripts")
    parser.add_argument("--no-hesse", action="store_false", default=True, dest="hesse")
    parser.add_argument("--no-frac", action="store_false", default=True, dest="frac")
    parser.add_argument("--no-GPU", action="store_false", default=True, dest="has_gpu")
    parser.add_argument("--method", default="BFGS", dest="method")
    results = parser.parse_args()
    if results.has_gpu:
        with tf.device("/device:GPU:0"):
            fit(method=results.method, hesse=results.hesse, frac=results.frac)
    else:
        with tf.device("/device:CPU:0"):
            fit(method=results.method, hesse=results.hesse, frac=results.frac)


if __name__ == "__main__":
    main()
