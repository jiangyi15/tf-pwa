import tensorflow as tf
import numpy as np
import json
from tf_pwa.utils import load_config_file, pprint, error_print
from fit_scipy_new import get_decay_chains, get_amplitude, load_params, cal_hesse_error
from tf_pwa.data import load_dat_file, data_to_tensor, split_generator
from tf_pwa.cal_angle import prepare_data_from_decay

# for fit
from tf_pwa.model import Model, FCN
from scipy.optimize import minimize
from tf_pwa.fitfractions import cal_fitfractions, cal_fitfractions_no_grad
import matplotlib.pyplot as plt
import time
import pickle


def prepare_data(fname, decs, particles=None, dtype="float64"):
    data_np = prepare_data_from_decay(fname, decs, particles=particles, dtype=dtype)
    return data_to_tensor(data_np)


from tf_pwa.applications import gen_data, gen_mc


def gen_data_from_mc():
    config_list = load_config_file("Resonances")
    decs, final_particles, decay = get_decay_chains(config_list)
    amp = get_amplitude(decs, config_list, decay)
    load_params(amp, "gen_params.json")
    gen_mc(4.59925172, [2.00698, 2.01028, 0.13957], 10000, "data/flat_mc.dat")
    gen_data(amp, final_particles, Ndata=10000, mcfile="data/flat_mc.dat", genfile="data/gen_data.dat",
             Poisson_fluc=True)


def fit():
    dtype = "float64"

    config_list = load_config_file("Resonances")
    decs, final_particles, decay = get_decay_chains(config_list)
    amp = get_amplitude(decs, config_list, decay)
    # data/pure1w.dat  data/flat_mc30w.dat
    data = prepare_data("data/data4600_new.dat", decs, particles=final_particles, dtype=dtype)
    mcdata = prepare_data("data/PHSP4600_new.dat", decs, particles=final_particles, dtype=dtype)
    bg = prepare_data("data/bg4600_new.dat", decs, particles=final_particles, dtype=dtype)
    w_bkg = 0.768331

    model = Model(amp, w_bkg=w_bkg)
    load_params(model.Amp, "glb_params_rp.json")
    model.Amp.vm.trans_params(True)
    args_name = model.Amp.vm.trainable_vars

    fcn = FCN(model, data, mcdata, bg, batch=65000)
    SCIPY = False
    hesse = True
    if SCIPY:
        def callback(x):
            print(fcn.cached_nll)

        f_g = fcn.model.Amp.vm.trans_fcn_grad(fcn.nll_grad)
        now = time.time()
        s = minimize(f_g, np.array(fcn.model.Amp.vm.get_all_val(True)), method="BFGS", jac=True, callback=callback,
                     options={"disp": 1, "gtol": 1e-4, "maxiter": 1000})
        model.Amp.vm.trans_params(True)
        xn = fcn.model.Amp.vm.get_all_val()
        print("########## fit state:")
        print(s)
        print("\nTime for fitting:", time.time() - now)
        val = {k: v.numpy() for k, v in model.Amp.variables.items()}
        err = {}
        if hesse:
            inv_he = cal_hesse_error(model.Amp, val, w_bkg, data, mcdata, bg, args_name, batch=20000)
            diag_he = inv_he.diagonal()
            hesse_error = np.sqrt(np.fabs(diag_he)).tolist()
            print(hesse_error)
            err = dict(zip(model.Amp.vm.trainable_vars, hesse_error))

    else:
        from tf_pwa.fit import fit_minuit
        m = fit_minuit(fcn, hesse=False)
        model.Amp.vm.trans_params(True)
        err_mtrx = m.np_covariance()
        np.save("error_matrix.npy", err_mtrx)
        val = dict(m.values)
        err = dict(m.errors)

    print("\n########## fit results:")
    for i in val:
        print("  ", i, ":", error_print(val[i], err.get(i, None)))

    frac = True
    err_frac = {}
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

    outdic = {"value": val, "error": err, "config": config_list,
              "frac": frac, "err_frac": err_frac, "NLL": fcn.cached_nll.numpy()}
    with open("glb_params_rp.json", "w") as f:
        json.dump(outdic, f, indent=2)


from tf_pwa.applications import fit_fractions
def gen_toy_sample():
    Ndata = 10000  # 8065
    Nbg = 0  # 3445
    w_bkg = 0  # .768331
    mcfile = "data/flat_mc30w.dat"
    bgfile = "data/bg4600_new.dat"
    params_file = "gen_params_rp.json"

    dtype = "float64"
    config_list = load_config_file("Resonances")
    decs, final_particles, decay = get_decay_chains(config_list)
    amp = get_amplitude(decs, config_list, decay)

    if w_bkg == 0:
        bg = None
        bgfile = None
    else:
        bg = prepare_data(bgfile, decs, particles=final_particles, dtype=dtype)
    mcdata = prepare_data(mcfile, decs, particles=final_particles, dtype=dtype)
    model = Model(amp, w_bkg=w_bkg)
    args_name = model.Amp.vm.trainable_vars

    var_arr = []
    err_arr = []
    frac_arr = []
    err_frac_arr = []
    for number in range(200):
        model.Amp.vm.trans_params(True)  # switch to rp

        load_params(model.Amp, params_file)
        # model.Amp.vm.trans_params(True)
        # gen_mc(4.59925172,[2.00698,2.01028,0.13957],60000,"data/flat_mc.dat")
        data = gen_data(model.Amp, final_particles, Ndata, mcfile=mcfile,
                        Poisson_fluc=True, Nbg=Nbg, wbg=w_bkg, bgfile=bgfile)

        fcn = FCN(model, data, mcdata, bg=bg, batch=65000)

        def callback(x):
            print(fcn.cached_nll)

        # fcn.model.Amp.vm.set_bound(bounds_dict)
        f_g = fcn.model.Amp.vm.trans_fcn_grad(fcn.nll_grad)
        now = time.time()
        s = minimize(f_g, np.array(fcn.model.Amp.vm.get_all_val(True)), method="BFGS", jac=True, callback=callback,
                     options={"disp": 1, "gtol": 1e-4, "maxiter": 1000})
        fcn.model.Amp.vm.set_all(s.x)
        model.Amp.vm.trans_params(False)  # switch to xy
        var = fcn.model.Amp.vm.get_all_dic(trainable_only=True)
        var_all = {k: v.numpy() for k, v in model.Amp.variables.items()}
        print("\nTime for fitting:", time.time() - now)
        print("fitting parameters:\n", var)

        err = {}
        inv_he = cal_hesse_error(model.Amp, var_all, w_bkg, data, mcdata, bg, args_name, batch=20000)
        diag_he = inv_he.diagonal()
        hesse_error = np.sqrt(np.fabs(diag_he)).tolist()
        err = dict(zip(model.Amp.vm.trainable_vars, hesse_error))
        print("errors:\n", err)

        frac, err_frac = fit_fractions(model, mcdata, inv_he, hesse=True)
        print("fitfractions:\n", frac)
        print("errors:\n", err_frac)

        var_arr.append(var)
        err_arr.append(err)
        frac_arr.append(frac)
        err_frac_arr.append(err_frac)
        if not s.success:
            print("NOT SUCCESSFUL FITTING")
        print("\n{0}END in {1}s\n".format(number, time.time() - now))

    print("$$$$$ var\n", var_arr)
    print("$$$$$ err\n", err_arr)
    print("$$$$$ frac\n", frac_arr)
    print("$$$$$ err_frac\n", err_frac_arr)

    output = open('toyMar9.pkl', 'wb')
    pickle.dump({"var": var_arr, "err": err_arr, "frac": frac_arr, "err_frac": err_frac_arr}, output, -1)
    output.close()


from tf_pwa.applications import compare_result


def compare_toy():
    with open("toyMar9.pkl", "rb") as f:
        dd = pickle.load(f)
    var = dd["var"]  # [{var_name:var_val}]
    err = dd["err"]
    frac = dd["frac"]  # [{frac_name:frac_val}]
    err_frac = dd["err_frac"]
    with open("gen_params_xy.json") as f:
        result = json.load(f)

    POLAR = False
    if POLAR:
        periodic_vars = []
        for v in prm["error"]:
            if v[-1] == "i":
                periodic_vars.append(v)
    else:
        periodic_vars = ["A->D2_2460p.BD2_2460p->D.C_totali", "A->D1_2420.DD1_2420->B.C_totali",
                         "A->D1_2420p.BD1_2420p->D.C_totali", "A->D1_2430.DD1_2430->B.C_totali",
                         "A->D1_2430p.BD1_2430p->D.C_totali"]

    for i in range(var.__len__()):
        print("##### Pull", i)
        compare_result(var[i], result["value"], err[i],  # result["error"],
                       figname="fig/tmp/var_pull_{}".format(i), yrange=8, periodic_vars=periodic_vars)
        """compare_result(result["frac"], frac[i], result["err_frac"], err_frac[i],
                       figname="fig/tmp/frac_pull_{}".format(i), yrange=5)"""


from tf_pwa.applications import plot_pull


def toy_pull():
    with open("toyMar10.pkl", "rb") as f:
        dd = pickle.load(f)
    var = dd["var"]  # [{var_name:var_val}]
    err = dd["err"]
    frac = dd["frac"]  # [{frac_name:frac_val}]
    err_frac = dd["err_frac"]
    with open("gen_params_xy.json") as f:
        result = json.load(f)
    del_list = []
    var = np.delete(var, del_list, axis=0)
    err = np.delete(err, del_list, axis=0)
    frac = np.delete(frac, del_list, axis=0)
    err_frac = np.delete(err_frac, del_list, axis=0)

    vv = {}
    err_vv = {}
    for i in var[0]:
        vv[i] = []
        err_vv[i] = []
    for i in vv:
        for j in var:
            vv[i].append(j[i])
        for j in err:
            err_vv[i].append(j[i])
    var = vv  # {var_name:[var_val]}
    err = err_vv

    ff = {}
    err_ff = {}
    for i in frac[0]:
        ff[i] = []
        err_ff[i] = []
    for i in ff:
        for j in frac:
            ff[i].append(j[i])
        for j in err_frac:
            err_ff[i].append(j[i])
    frac = ff  # {frac_name:[frac_val]}
    err_frac = err_ff

    m = []
    s = []
    e_m = []
    e_s = []
    for i in var:
        mean, sigma, err_m, err_s = plot_pull(var[i], "var" + i, norm=True, value=result["value"][i], error=err[i])
        m.append(mean)
        s.append(sigma)
        e_m.append(err_m)
        e_s.append(err_s)
        print("Plot pull of {}".format(i))
    print("mean", m)
    print("err_mean", e_m)
    print("sigma", s)
    print("err_sigma", e_s)

    m = []
    s = []
    e_m = []
    e_s = []
    for i in frac:
        mean, sigma, err_m, err_s = plot_pull(frac[i], "frac" + i, norm=True, value=result["frac"][i], error=err_frac[i])
        m.append(mean)
        s.append(sigma)
        e_m.append(err_m)
        e_s.append(err_s)
        print("Plot pull of {}".format(i))
    print("mean", m)
    print("err_mean", e_m)
    print("sigma", s)
    print("err_sigma", e_s)


def draw_errorbar():
    mu = []
    err_mu = []
    plt.errorbar(range(len(mu)),mu,yerr=err_mu,fmt="oy",ecolor='r',elinewidth=1,capsize=3)
    plt.errorbar(range(len(mu)),0,fmt='g')
    plt.xlabel("var index")
    plt.ylabel("value")
    plt.title("mu")
    plt.show()
    #plt.savefig("errbar_mu")
    #plt.clf()

    sigma = []
    err_sigma = []
    plt.errorbar(range(len(sigma)),sigma,yerr=err_sigma,fmt="oy",ecolor='r',elinewidth=1,capsize=3)
    plt.errorbar(range(len(sigma)),1,fmt='g')
    plt.xlabel("var index")
    plt.ylabel("value")
    plt.title("sigma")
    plt.show()
    #plt.savefig("errbar_sigma")
    #plt.clf()


if __name__ == "__main__":
    toy_pull()
