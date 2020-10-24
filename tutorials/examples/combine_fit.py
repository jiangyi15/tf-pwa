#!/usr/bin/env python3

import datetime

# from pprint import pprint
import json
import os.path
import sys
import time

import numpy as np
from scipy.optimize import minimize

import tf_pwa
from tf_pwa.amp import AmplitudeModel, DecayGroup, HelicityDecay, Particle
from tf_pwa.cal_angle import prepare_data_from_decay
from tf_pwa.data import data_to_tensor, split_generator
from tf_pwa.fitfractions import cal_fitfractions
from tf_pwa.model import FCN, CombineFCN, Model
from tf_pwa.tensorflow_wrapper import tf
from tf_pwa.utils import error_print, load_config_file, pprint
from tf_pwa.variable import VarsManager

this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + "/..")


# log_dir = "./cached_dir/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


data_file_name = [
    ["./data/data4600_new.dat", "data/Dst0_data4600_new.dat"],
    ["./data/bg4600_new.dat", "data/Dst0_bg4600_new.dat"],
    ["./data/PHSP4600_new.dat", "data/Dst0_PHSP4600_new.dat"],
]

data_file_name2 = [
    ["./data/data4600_new.dat", "data/Dst0_data4600_new.dat"],
    ["./data/bg4600_new.dat", "data/Dst0_bg4600_new.dat"],
    ["./data/PHSP4600_new.dat", "data/Dst0_PHSP4600_new.dat"],
]


def prepare_data(fnames, decs, particles=None, dtype="float64"):
    tname = ["data", "bg", "PHSP"]
    data_np = {}
    for i, name in enumerate(fnames):
        data_np[tname[i]] = prepare_data_from_decay(
            name[0], decs, particles=particles, dtype=dtype
        )
    data, bg, mcdata = data_to_tensor([data_np[i] for i in tname])
    return data, bg, mcdata


def cal_hesse_error(amp, val, w_bkg, data, mcdata, bg, args_name, batch):
    a_h = FCN(Model(amp, w_bkg), data, mcdata, bg=bg, batch=batch)
    t = time.time()
    nll, g, h = a_h.nll_grad_hessian(
        val
    )  # data_w,mcdata,weight=weights,batch=50000)
    print("Time for calculating errors:", time.time() - t)
    inv_he = np.linalg.pinv(h.numpy())
    np.save("error_matrix.npy", inv_he)
    return inv_he


def get_decay_chains(config_list):
    a = Particle("A", J=1, P=-1, spins=(-1, 1))
    b = Particle("B", J=1, P=-1)
    c = Particle("C", J=0, P=-1)
    d = Particle("D", J=1, P=-1)
    decay = {}
    for i in config_list:
        config = config_list[i]
        res = Particle(
            i,
            config["J"],
            config["Par"],
            mass=config["m0"],
            width=config["g0"],
        )
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
    return decs, [d, b, c], decay


def get_amplitude(decs, config_list, decay, polar=True, vm=None):
    amp = AmplitudeModel(decs, polar, vm=vm)
    for i in config_list:
        if "coef_head" in config_list[i]:
            coef_head = config_list[i]["coef_head"]
            for a, b in zip(decay[i], decay[coef_head]):
                b.g_ls.sameas(a.g_ls)
            for j in decs.chains:
                if decay[i][0] in j:
                    for k in decs.chains:
                        if decay[coef_head][0] in k:
                            k.total.r_shareto(j.total)
        if "total" in config_list[i]:
            for j in decs.chains:
                if decay[i][0] in j:
                    j.total.fixed(config_list[i]["total"])
        if ("float" in config_list[i]) and config_list[i]["float"]:
            if "m" in config_list[i]["float"]:
                decay[i][1].core.mass.freed()
            if "g" in config_list[i]["float"]:
                decay[i][1].core.width.freed()
    return amp


def load_params(amp, init_params):
    try:
        with open(init_params) as f:
            param_0 = json.load(f)
            if "value" in param_0:
                param_0 = param_0["value"]
            param = {}
            for k, v in param_0.items():
                if k.endswith(":0"):
                    k = k[:-2]
                param[k] = v
            print("using {}".format(init_params))
            amp.set_params(param)
        RDM_INI = False
    except Exception as e:
        print(e)
        RDM_INI = True
        print("using RANDOM parameters")


def fit(method="BFGS", init_params="init_params.json", hesse=True, frac=True):
    POLAR = True  # fit in polar coordinates. should be consistent with init_params.json if any
    GEN_TOY = False  # use toy data (mcdata and bg stay the same). REMEMBER to update gen_params.json

    dtype = "float64"
    w_bkg = 0.768331
    # set_gpu_mem_growth()
    tf.keras.backend.set_floatx(dtype)
    # open Resonances list as dict
    config_list = load_config_file("Resonances")
    config_list2 = load_config_file("Resonances")

    vm = VarsManager()

    decs, final_particles, decay = get_decay_chains(config_list)
    data, bg, mcdata = prepare_data(
        data_file_name, decs, particles=final_particles, dtype=dtype
    )
    decs2, final_particles2, decay2 = get_decay_chains(config_list)
    data2, bg2, mcdata2 = prepare_data(
        data_file_name2, decs2, particles=final_particles2, dtype=dtype
    )

    amp = get_amplitude(decs, config_list, decay, polar=POLAR, vm=vm)
    amp2 = get_amplitude(decs2, config_list2, decay2, polar=POLAR, vm=vm)

    load_params(amp, init_params)
    load_params(amp2, init_params)

    model = Model(amp, w_bkg=w_bkg)
    model2 = Model(amp2, w_bkg=w_bkg)
    pprint(model.get_params())

    fcn = CombineFCN(
        [model, model2],
        [data, data2],
        [mcdata, mcdata2],
        bg=[bg, bg2],
        batch=65000,
    )

    # fit configure
    args = {}
    args_name = vm.trainable_vars
    x0 = []
    bnds = []
    bounds_dict = {
        # "Zc_4160_m:0":(4.1,4.22),
        # "Zc_4160_g:0":(0,None)
    }

    for i in model.Amp.trainable_variables:
        args[i.name] = i.numpy()
        x0.append(i.numpy())
        if i.name in bounds_dict:
            bnds.append(bounds_dict[i.name])
        else:
            bnds.append((None, None))
        args["error_" + i.name] = 0.1

    points = []
    nlls = []
    now = time.time()
    maxiter = 1000

    method = "BFGS"

    def callback(x):
        if np.fabs(x).sum() > 1e7:
            x_p = dict(zip(args_name, x))
            raise Exception("x too large: {}".format(x_p))
        points.append(vm.get_all_val())
        nlls.append(float(fcn.cached_nll))
        print(fcn.cached_nll)

    vm.set_bound(bounds_dict)
    f_g = vm.trans_fcn_grad(fcn.nll_grad)
    s = minimize(
        f_g,
        np.array(vm.get_all_val(True)),
        method=method,
        jac=True,
        callback=callback,
        options={"disp": 1, "gtol": 1e-4, "maxiter": maxiter},
    )
    xn = vm.get_all_val()
    print("########## fit state:")
    print(s)
    print("\nTime for fitting:", time.time() - now)
    model.Amp.vm.trans_params(POLAR)
    val = {k: v.numpy() for k, v in model.Amp.variables.items()}
    with open("final_params.json", "w") as f:
        json.dump({"value": val}, f, indent=2)

    err = {}
    inv_he = cal_hesse_error(
        model.Amp, val, w_bkg, data, mcdata, bg, args_name, batch=20000
    )
    diag_he = inv_he.diagonal()
    hesse_error = np.sqrt(np.fabs(diag_he)).tolist()
    print(hesse_error)
    err = dict(zip(model.Amp.vm.trainable_vars, hesse_error))
    print("\n########## fit results:")
    for i in val:
        print("  ", i, ":", error_print(val[i], err.get(i, None)))

    outdic = {"value": val, "error": err, "config": config_list}
    with open("final_params.json", "w") as f:
        json.dump(outdic, f, indent=2)

    err_frac = {}

    frac, grad = cal_fitfractions(
        model.Amp, list(split_generator(mcdata, 25000))
    )

    for i in frac:
        err_frac[i] = np.sqrt(np.dot(np.dot(inv_he, grad[i]), grad[i]))

    print("########## fit fractions")
    for i in frac:
        print(i, ":", error_print(frac[i], err_frac.get(i, None)))
    print("\nEND\n")
    """outdic = {"value": val, "error": err, "frac": frac, "err_frac": err_frac}
    with open("glbmin_params.json", "w") as f:
        json.dump(outdic, f, indent=2)
    """


def main():
    import argparse

    parser = argparse.ArgumentParser(description="simple fit scripts")
    parser.add_argument(
        "--no-GPU", action="store_false", default=True, dest="has_gpu"
    )
    parser.add_argument("--method", default="BFGS", dest="method")
    results = parser.parse_args()
    if results.has_gpu:
        with tf.device("/device:GPU:0"):
            fit(method=results.method)
    else:
        with tf.device("/device:CPU:0"):
            fit(method=results.method)


if __name__ == "__main__":

    # summary_writer = tf.summary.create_file_writer(log_dir)
    # with summary_writer.as_default():
    main()
