#!/usr/bin/env python3
from tf_pwa.model_new import Model, FCN
from tf_pwa.tensorflow_wrapper import tf
import time
import numpy as np
from pprint import pprint
import json
import os
import datetime
from scipy.optimize import minimize, BFGS, basinhopping
import tf_pwa
from tf_pwa.utils import load_config_file, flatten_np_data, pprint, error_print, std_polar
from tf_pwa.fitfractions import cal_fitfractions, cal_fitfractions_no_grad
import math
# from tf_pwa.bounds import Bounds

from tf_pwa.amp import AmplitudeModel, DecayGroup, HelicityDecay, Particle, get_name

from tf_pwa.data import data_to_numpy, data_to_tensor, split_generator
from tf_pwa.cal_angle import prepare_data_from_decay

log_dir = "./cached_dir/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 


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
        cached_data = data_to_tensor(cached_data)
        pprint(cached_data)
        data = cached_data["data"]
        bg = cached_data["bg"]
        mcdata = cached_data["PHSP"]
        print("using cached data")
        return data, bg, mcdata
    data_np = {}
    for i, name in enumerate(fname):
        data_np[tname[i]] = prepare_data_from_decay(name[0], decs, particles=particles, dtype=dtype)

    data, bg, mcdata = data_to_tensor([data_np[i] for i in tname])
    #import pprint
    #pprint.pprint(data)
    save_cached_data(data_to_numpy({"data": data, "bg": bg, "PHSP": mcdata}))
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


def get_decay_chains(config_list):
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
    return decs, [d, b, c], decay


def get_amplitude(decs, config_list, decay):
    amp = AmplitudeModel(decs)

    def coef_combine(a, b, r="r", g_ls="g_ls"):
        name_a = get_name(a, g_ls)+r
        name_b = get_name(b, g_ls)+r
        if name_b in amp.vm.variables:
            amp.vm.variables[name_a] = amp.vm.variables[name_b]
        else:
            print("not found,", name_b)
        if name_a in amp.vm.trainable_vars:
            amp.vm.trainable_vars.remove(name_a)

    # print(amp.vm.variables)
    # print(amp.vm.trainable_vars)
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
                        coef_combine(a, b, "_"+str(j)+"r")
                        coef_combine(a, b, "_"+str(j)+"i")
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
    # print(amp.vm.trainable_vars)
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

    decs, final_particles, decay = get_decay_chains(config_list)
    data, bg, mcdata = prepare_data(decs, particles=final_particles, dtype=dtype)
    amp = get_amplitude(decs, config_list, decay)
    load_params(amp, init_params)

    # print(amp.vm.variables)
    model = Model(amp, w_bkg=w_bkg)
    # print(model.Amp(data))
    # tf.summary.trace_on(graph=True, profiler = True)
    now = time.time()
    nll = model.nll(data, mcdata, bg=bg)
    print(nll)
    print(time.time() - now)
    # tf.summary.trace_export(name="sum_amp", step=0, profiler_outdir=log_dir)
    now = time.time()
    print(model.nll_grad(data, mcdata, bg=bg, batch=15000))
    print(time.time() - now)
    # now = time.time()
    # print(model.nll_grad_hessian(data, mcdata, batch=10000))
    # print(time.time() - now)
    fcn = FCN(model, data, mcdata, bg=bg, batch=65000)

    # fit configure
    args = {}
    args_name = model.Amp.vm.trainable_vars
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

    check_grad = False
    if check_grad:
        _, gs0 = fcn.nll_grad(x0)
        gs = []
        for i, name in enumerate(args_name):
            x0[i] += 1e-5
            nll0, _ = fcn.nll_grad(x0)
            x0[i] -= 2e-5
            nll1, _ = fcn.nll_grad(x0)
            x0[i] += 1e-5
            gs.append((nll0-nll1)/2e-5)
            print(gs[i], gs0[i])
            
    check_hessian = False
    if check_hessian:
        nll, g, hs0 = fcn.nll_grad_hessian(x0, 20000)
        hs = []
        for i, name in enumerate(args_name):
            x0[i] += 1e-5
            _, gi1 = fcn.nll_grad(x0)
            x0[i] -= 2e-5
            _, gi2 = fcn.nll_grad(x0)
            x0[i] += 1e-5
            hs.append((gi1 - gi2)/2e-5)
            print(hs[i], hs0[i])

    points = []
    nlls = []
    now = time.time()
    maxiter = 1  ##000

    # s = basinhopping(f.nll_grad,np.array(x0),niter=6,disp=True,minimizer_kwargs={"jac":True,"options":{"disp":True}})
    if method in ["BFGS", "CG", "Nelder-Mead"]:
        def callback(x):
            if np.fabs(x).sum() > 1e7:
                x_p = dict(zip(args_name, x))
                raise Exception("x too large: {}".format(x_p))
            points.append(model.Amp.vm.get_all_val())
            nlls.append(float(fcn.cached_nll))
            if len(nlls) > maxiter:
                with open("fit_curve.json", "w") as f:
                    json.dump({"points": points, "nlls": nlls}, f, indent=2)
                pass  # raise Exception("Reached the largest iterations: {}".format(maxiter))
            print(fcn.cached_nll)

        #bd = Bounds(bnds)
        fcn.model.Amp.vm.set_bound(bounds_dict)
        f_g = fcn.model.Amp.vm.trans_fcn_grad(fcn.nll_grad)
        s = minimize(f_g, np.array(fcn.model.Amp.vm.get_all_val(True)), method=method, jac=True, callback=callback, options={"disp": 1})
        xn = fcn.model.Amp.vm.get_all_val() #bd.get_y(s.x)
    elif method in ["L-BFGS-B"]:
        def callback(x):
            if np.fabs(x).sum() > 1e7:
                x_p = dict(zip(args_name, x))
                raise Exception("x too large: {}".format(x_p))
            points.append([float(i) for i in x])
            nlls.append(float(fcn.cached_nll))

        s = minimize(fcn.nll_grad, np.array(x0), method=method, jac=True, bounds=bnds, callback=callback,
                     options={"disp": 1, "maxcor": 10000, "ftol": 1e-15, "maxiter": maxiter})
        xn = s.x
    else:
        pass  # raise Exception("unknown method")
    print("########## fit state:")
    # print(s)
    print("\nTime for fitting:", time.time() - now)
    val = {k: v.numpy() for k, v in model.Amp.variables.items()}
    with open("final_params.json", "w") as f:
        json.dump({"value": val}, f, indent=2)
    err = {}
    if hesse:
        inv_he = cal_hesse_error(model.Amp, val, w_bkg, data, mcdata, bg, args_name, batch=20000)
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
    # print("\n########## ratios of partial wave amplitude square")
    # calPWratio(params,POLAR)

    if frac:
        err_frac = {}
        if hesse:
            frac, grad = cal_fitfractions(model.Amp, list(split_generator(mcdata, 25000)))
        else:
            frac = cal_fitfractions_no_grad(model.Amp, list(split_generator(mcdata, 45000)))

        for i in frac:
            if hesse:
                err_frac[i] = np.sqrt(np.dot(np.dot(inv_he, grad[i]), grad[i]))
        print("########## fit fractions")
        for i in frac:
            print(i, ":", error_print(frac[i], err_frac.get(i, None)))
    print("\nEND\n")




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
    
    # summary_writer = tf.summary.create_file_writer(log_dir)
    # with summary_writer.as_default():
    main()
