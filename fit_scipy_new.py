#!/usr/bin/env python3
from tf_pwa.model_new import Model, FCN
import tensorflow as tf
import time
import numpy as np
from pprint import pprint
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

from tf_pwa.amp import AmplitudeModel, DecayGroup, HelicityDecay, Particle, get_name

from tf_pwa.data import prepare_data_from_decay, data_to_numpy


def load_cached_data(cached_data_file="cached_data_new.npy"):
    cached_dir = "./cached_dir"
    if not os.path.exists(cached_dir):
        return None
    cached_path = os.path.join(cached_dir, cached_data_file)
    if os.path.exists(cached_path):
        cached_data = tf_pwa.load_data(cached_path)
        return cached_data
    return None


def save_cached_data(cached_data, cached_data_file="cached_data_new.npy"):
    cached_dir = "./cached_dir"
    if not os.path.exists(cached_dir):
        os.mkdir(cached_dir)
    cached_path = os.path.join(cached_dir, cached_data_file)
    tf_pwa.save_data(cached_path, cached_data)


def prepare_data(decs, particles=None, dtype="float64"):
    fname = [
        ["./data/data4600_new.dat", "data/Dst0_data4600_new.dat"],
        ["./data/bg4600_new.dat", "data/Dst0_bg4600_new.dat"],
        ["./data/PHSP4600_new.dat", "data/Dst0_PHSP4600_new.dat"]
    ]
    tname = ["data", "bg", "PHSP"]
    try:
        cached_data = load_cached_data()
    except Exception as e:
        print(e)
        cached_data = None
    if cached_data is not None:
        cached_data = data_to_numpy(cached_data)
        pprint(cached_data)
        data = cached_data["data"]
        bg = cached_data["bg"]
        mcdata = cached_data["PHSP"]
        print("using cached data")
        return data, bg, mcdata
    data_np = {}
    for i, name in enumerate(fname):
        data_np[tname[i]] = prepare_data_from_decay(name[0], decs, particles=particles, dtype=dtype)

    data, bg, mcdata = [data_np[i] for i in tname]
    #import pprint
    #pprint.pprint(data)
    save_cached_data({"data": data, "bg": bg, "PHSP": mcdata})
    return data, bg, mcdata


def cal_hesse_error(amp, val, w_bkg, data, mcdata, bg, args_name, batch):
    a_h = FCN(Model(amp, w_bkg), data, mcdata, bg=bg, batch=batch)
    t = time.time()
    nll, g, h = a_h.nll_grad_hessian(val)  # data_w,mcdata,weight=weights,batch=50000)
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
    decay = {}
    for i in config_list:
        config = config_list[i]
        res = Particle(i, config["J"], config["Par"], mass=config["m0"], width=config["g0"])
        chain = config["Chain"]
        if chain < 0:
            dec1 = HelicityDecay(a, [res, c])
            dec2 = HelicityDecay(res, [b, d])
        elif chain < 100:
            dec1 = HelicityDecay(a, [res, d])
            dec2 = HelicityDecay(res, [b, c])
        elif chain < 200:
            dec1 = HelicityDecay(a, [res, b])
            dec2 = HelicityDecay(res, [d, c])
        else:
            raise Exception("unknown chain")
        decay[i] = [dec1, dec2]

    decs = DecayGroup(a.chain_decay())
    data, bg, mcdata = prepare_data(decs, particles=[d, b, c], dtype=dtype)

    amp = AmplitudeModel(decs)

    def coef_combine(a, b, r="r", g_ls="g_ls"):
        name_a = get_name(a, g_ls)+r
        name_b = get_name(b, g_ls)+r
        if name_b in amp.vm.variables:
            amp.vm.variables[name_a] = amp.vm.variables[name_b]
        else:
            print(name_b)
        if name_a in amp.vm.trainable_vars:
            amp.vm.trainable_vars.remove(name_a)

    print(amp.vm.variables)
    print(amp.vm.trainable_vars)
    for i in config_list:
        if "coef_head" in config_list[i]:
            coef_head = config_list[i]["coef_head"]
            for a, b in zip(decay[i], decay[coef_head]):
                num = len(a.get_ls_list())
                if num == 1:
                    coef_combine(a, b)
                    coef_combine(a, b, "i")
                else:
                    for j in range(num):
                        coef_combine(a, b, str(j)+"r")
                        coef_combine(a, b, str(j)+"i")
            for j in decs:
                if decay[i][0] in j:
                    for k in decs:
                        if decay[coef_head][0] in k:
                            coef_combine(j, k, g_ls="total")
    for j in decs:
        name = get_name(j, "total")
        amp.vm.trainable_vars.remove(name+"r")
        amp.vm.trainable_vars.remove(name+"i")
        break
    print(amp.vm.trainable_vars)
    model = Model(amp, w_bkg=w_bkg)
    now = time.time()
    print(model.nll(data, mcdata))
    print(time.time() - now)
    now = time.time()
    print(model.nll_grad(data, mcdata, batch=65000))
    print(time.time() - now)
    # now = time.time()
    # print(model.nll_grad_hessian(data, mcdata, batch=10000))
    # print(time.time() - now)
    fcn = FCN(model, data, mcdata, bg=bg)


    # fit configure
    args = {}
    args_name = []
    x0 = []
    bnds = []
    bounds_dict = {
        # "Zc_4160_m:0":(4.1,4.22),
        # "Zc_4160_g:0":(0,None)
    }

    for i in model.Amp.trainable_variables:
        args[i.name] = i.numpy()
        x0.append(i.numpy())
        args_name.append(i.name)
        if i.name in bounds_dict:
            bnds.append(bounds_dict[i.name])
        else:
            bnds.append((None, None))
        args["error_" + i.name] = 0.1

    check_grad = False
    if check_grad:
        _, gs0 = fcn.nll_grad(x0)
        gs = []
        for i, name in enumerate(args_name):
            x0[i] += 1e-3
            nll0, _ = fcn.nll_grad(x0)
            x0[i] -= 2e-3
            nll1, _ = fcn.nll_grad(x0)
            x0[i] += 1e-3
            gs.append((nll0-nll1)/2e-3)
            print(gs[i], gs0[i])

    points = []
    nlls = []
    now = time.time()
    maxiter = 2000
    bd = Bounds(bnds)
    f_g = bd.trans_f_g(fcn.nll_grad)

    def callback(x):
        if np.fabs(x).sum() > 1e7:
            x_p = dict(zip(args_name, x))
            raise Exception("x too large: {}".format(x_p))
        point = [float(i) for i in bd.get_y(x)]
        print(dict(zip(args_name, point)))
        points.append(point)
        nlls.append(float(fcn.cached_nll))
        if len(nlls) > maxiter:
            with open("fit_curve.json", "w") as f:
                json.dump({"points": points, "nlls": nlls}, f, indent=2)
            raise Exception("Reached the largest iterations: {}".format(maxiter))
        print(fcn.cached_nll)

    s = minimize(f_g, np.array(bd.get_x(x0)), method=method, jac=True, callback=callback, options={"disp": 1})
    xn = bd.get_y(s.x)
    print(xn)


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
