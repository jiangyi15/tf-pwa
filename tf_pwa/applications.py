import numpy as np
import tensorflow as tf
import time

from .fitfractions import cal_fitfractions, cal_fitfractions_no_grad
def fit_fractions(model,mcdata,config_list,inv_he,hesse):
    if hesse:
        mcdata_cached = model.Amp.cache_data(*mcdata, batch=10000)
        frac, grad = cal_fitfractions(model.Amp, mcdata_cached, kwargs={"cached": True})
    else:
        mcdata_cached = model.Amp.cache_data(*mcdata, batch=65000)
        frac = cal_fitfractions_no_grad(model.Amp, mcdata_cached, kwargs={"cached": True})
    err_frac = {}
    for i in config_list:
        if hesse:
            err_frac[i] = np.sqrt(np.dot(np.dot(inv_he, grad[i]), grad[i]))
        else:
            err_frac[i] = None
    return frac, err_frac


def error_matrix():
    pass


def calPWratio(params, POLAR=True):
    dtype = "float64"
    w_bkg = 0.768331
    # set_gpu_mem_growth()
    tf.keras.backend.set_floatx(dtype)
    config_list = load_config_file("Resonances")
    a = Model(config_list, w_bkg, kwargs={"polar": POLAR})

    args_name = []
    for i in a.Amp.trainable_variables:
        args_name.append(i.name)
    # a.Amp.polar=True

    a.set_params(params)
    if not POLAR:  # if final_params.json is not in polar coordinates
        i = 0
        for v in args_name:
            if len(v) > 15:
                if i % 2 == 0:
                    tmp_name = v
                    tmp_val = params[v]
                else:
                    params[tmp_name] = np.sqrt(tmp_val ** 2 + params[v] ** 2)
                    params[v] = np.arctan2(params[v], tmp_val)
            i += 1
        a.set_params(params)
    fname = [["data/data4600_new.dat", "data/Dst0_data4600_new.dat"],
             ["data/bg4600_new.dat", "data/Dst0_bg4600_new.dat"],
             ["data/PHSP4600_new.dat", "data/Dst0_PHSP4600_new.dat"]
             ]
    tname = ["data", "bg", "PHSP"]
    data_np = {}
    for i in range(3):
        data_np[tname[i]] = cal_ang_file(fname[i][0], dtype)

    def load_data(name):
        dat = []
        tmp = flatten_np_data(data_np[name])
        for i in param_list:
            tmp_data = tf.Variable(tmp[i], name=i, dtype=dtype)
            dat.append(tmp_data)
        return dat

    data = load_data("data")
    bg = load_data("bg")
    mcdata = load_data("PHSP")
    data_cache = a.Amp.cache_data(*data)
    bg_cache = a.Amp.cache_data(*bg)
    mcdata_cache = a.Amp.cache_data(*mcdata)
    total = a.Amp(mcdata_cache, cached=True)
    a_sig = {}

    # res_list = [[i] for i in config_list]
    res_list = [
        ["Zc_4025"],
        ["Zc_4160"],
        ["D1_2420", "D1_2420p"],
        ["D1_2430", "D1_2430p"],
        ["D2_2460", "D2_2460p"],
    ]

    config_res = [part_config(config_list, i) for i in res_list]
    PWamp = {}
    for i in range(len(res_list)):
        name = res_list[i]
        if isinstance(name, list):
            if len(name) > 1:
                name = reduce(lambda x, y: "{}+{}".format(x, y), res_list[i])
            else:
                name = name[0]
        a_sig[i] = Model(config_res[i], w_bkg, kwargs={"polar": POLAR})
        p_list = [[], []]
        for p in a_sig[i].get_params():
            if p[-3] == 'r' and len(p) > 15:
                if p[8] == 'd':
                    p_list[1].append(p)
                else:
                    p_list[0].append(p)
        first = True
        for p in p_list[0]:
            a_sig[i].set_params(params)
            for q in p_list[0]:
                a_sig[i].set_params({q: 0})
            a_sig[i].set_params({p: params[p]})
            if first:
                norm = a_sig[i].Amp(mcdata_cache, cached=True).numpy().sum()
                print(p[:-3], "\t", 1.0)
                first = False
            else:
                print(p[:-3], "\t", a_sig[i].Amp(mcdata_cache, cached=True).numpy().sum() / norm)
        first = True
        for p in p_list[1]:
            a_sig[i].set_params(params)
            for q in p_list[1]:
                a_sig[i].set_params({q: 0})
            a_sig[i].set_params({p: params[p]})
            if first:
                norm = a_sig[i].Amp(mcdata_cache, cached=True).numpy().sum()
                print(p[:-3], "\t", 1.0)
                first = False
            else:
                print(p[:-3], "\t", a_sig[i].Amp(mcdata_cache, cached=True).numpy().sum() / norm)
        print()
        # print(a_sig[i].get_params())
        # a_weight[i] = a_sig[i].Amp(mcdata_cache,cached=True).numpy()
        # PWamp[name] = a_weight[i].sum()/(n_data - w_bkg*n_bg)


def cal_hesse_error(model):
    t = time.time()
    nll,g,h = model.cal_nll_hessian()#data_w,mcdata,weight=weights,batch=50000)
    print("Time for calculating errors:",time.time()-t)
    inv_he = np.linalg.pinv(h.numpy())
    np.save("error_matrix.npy",inv_he)
    return inv_he
