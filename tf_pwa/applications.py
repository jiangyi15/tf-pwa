import numpy as np
import tensorflow as tf
import time

from .fitfractions import cal_fitfractions, cal_fitfractions_no_grad


def fit_fractions(model, mcdata, config_list, inv_he, hesse):
    '''
    calculate fit fractions of all resonances
    :param model:
    :param mcdata:
    :param config_list:
    :param inv_he:
    :param hesse:
    :return:
    '''
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


def corr_coef_matrix(npy_name):
    '''
    obtain correlation coefficients matrix of all trainable variables
    :param npy_name:
    :return:
    '''
    err_mtx = np.load(npy_name)
    err = np.sqrt(err_mtx.diagonal())
    diag_mtx = np.diag(1 / err)
    tmp_mtx = np.matmul(diag_mtx, err_mtx)
    cc_mtx = np.matmul(tmp_mtx, diag_mtx)
    return cc_mtx


def calPWratio(params, POLAR=True):
    '''
    calculate the ratio of different partial waves
    :param params:
    :param POLAR:
    :return:
    '''
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
    '''
    calculate the hesse error of all trainable variables
    :param model:
    :return:
    '''
    t = time.time()
    nll, g, h = model.cal_nll_hessian()  # data_w,mcdata,weight=weights,batch=50000)
    print("Time for calculating errors:", time.time() - t)
    inv_he = np.linalg.pinv(h.numpy())
    np.save("error_matrix.npy", inv_he)
    return inv_he


from .data import load_dat_file, data_to_tensor
from tf_pwa.cal_angle import prepare_data_from_decay


def gen_data(amp, particles, Ndata, mcfile, Nbg=0, wbg=0, Poisson_fluc=False,
             bgfile=None, genfile="data/gen_data.dat"):
    Nbg = round(wbg * Nbg)
    Nmc = Ndata - Nbg  # 8065-3445*0.768331
    if Poisson_fluc:  # Poisson
        Nmc = np.random.poisson(Nmc)
        Nbg = np.random.poisson(Nbg)
    print("data:", Nmc + Nbg, ", sig:", Nmc, ", bkg:", Nbg)
    dtype = "float64"

    phsp = prepare_data_from_decay(mcfile, amp.decay_group, particles=particles, dtype=dtype)
    phsp = data_to_tensor(phsp)
    ampsq = amp(phsp)
    ampsq_max = tf.reduce_max(ampsq).numpy()
    Nsample = ampsq.__len__()
    n = 0
    idx_list = []

    while n < Nmc:
        uni_rdm = tf.random.uniform([Nsample], minval=0, maxval=ampsq_max, dtype=dtype)
        list_rdm = tf.random.uniform([Nsample], dtype=tf.int64, maxval=Nsample)
        j = 0
        for i in list_rdm:
            if ampsq[i] > uni_rdm[j]:
                idx_list.append(i)
                n += 1
            j += 1
            if n == Nmc:
                break
    idx_list = tf.stack(idx_list).numpy()

    data_tmp = load_dat_file(mcfile, particles, dtype)
    for i in particles:
        data_tmp[i] = np.array(data_tmp[i])[idx_list]
    data_gen = []

    if Nbg:
        bg_tmp = load_dat_file(bgfile, particles, dtype)
        bg_idx = tf.random.uniform([Nbg], dtype=tf.int64,
                                   maxval=len(bg_tmp[particles[0]]))  # np.random.randint(len(bg),size=Nbg)
        bg_idx = tf.stack(bg_idx).numpy()
        for i in particles:
            tmp = bg_tmp[i][bg_idx]
            data_tmp[i] = np.append(data_tmp[i], tmp, axis=0)
            data_gen.append(data_tmp[i])
    else:
        for i in particles:
            data_gen.append(data_tmp[i])

    data_gen = np.transpose(data_gen, [1, 0, 2])
    np.random.shuffle(data_gen)
    data_gen = data_gen.reshape(-1, 4)
    np.savetxt(genfile, data_gen)

    data = prepare_data_from_decay(genfile, amp.decay_group, particles=particles, dtype=dtype)
    return data_to_tensor(data)


from .phasespace_tf import PhaseSpaceGenerator


def gen_mc(mother, daughters, number, outfile="data/flat_mc.dat"):
    '''
    generate phase space MC data not considering the effect of detector performance
    :param mother: 4.59925172
    :param daughters: [2.00698,2.01028,0.13957] DBC: D*0 D*- pi+
    :param number:
    :param outfile:
    :return:
    '''
    phsp = PhaseSpaceGenerator(mother, daughters)
    flat_mc_data = phsp.generate(number)
    pf = []
    for i in len(daughters):
        p = flat_mc_data[i]
        p_a = np.array([p.T, p.X, p.Y, p.Z]).reshape((4, -1))  # (T,X,Y,Z)
        pf.append(p_a)  # daughters和pf的顺序须一致
    pf = np.transpose(pf, (2, 0, 1)).reshape((-1, 4))
    np.savetxt(outfile, pf)  # 一个不包含探测器效率的MC样本
    return pf


from .fit import fit_scipy, fit_minuit, fit_multinest


def fit(Use="scipy", **kwargs):
    '''
    fit using scipy, iminuit or pymultinest
    :param Use:
    :param kwargs:
    :return:
    '''
    if Use == "scipy":
        ret = fit_scipy(**kwargs)
    elif Use == "minuit":
        ret = fit_minuit(**kwargs)
    elif Use == "multinest":
        ret = fit_multinest(**kwargs)
    else:
        raise Exception("Unknown fit tool {}".format(Use))
    return ret


from .significance import significance


def cal_significance():
    pass


### plot-related ###
import matplotlib.pyplot as plt
from scipy.stats import norm as Norm


def plot_pull(data, name, nbins=20, norm=False, value=None, error=None):
    data = np.array(data)
    if norm:
        if value == None or error == None:
            raise Exception("Need value or error for normed pull!")
        data = (data - value) / error

    n, bins, patches = plt.hist(data, nbins, density=1, alpha=0.6)
    mean, sigma = Norm.fit(data)
    y = Norm.pdf(bins, mean, sigma)
    plt.plot(bins, y, "r*-")
    plt.xlabel(name)
    plt.title(name + ": mean=%.3f, sigma=%.3f" % (mean, sigma))
    plt.savefig("fig/" + name + "_pull.png")
    plt.clf()
    return mean, sigma


def likelihood_profile(var_name, start=None, end=None, step=None, values=None, errors=None, mode="bothsides"):
    if start == None or end == None:
        x_mean = values[var_name]
        x_sigma = errors[var_name]
        start = x_mean - 10 * x_sigma
        end = x_mean + 10 * x_sigma
    else:
        x_mean = (end + start) / 2
    if step == None:
        step = (end - start) / 100
    if mode == "bothsides":
        x1 = np.arange(x_mean, start, -step)
        x2 = np.arange(x_mean, end, step)
        #
    elif mode == "back&forth":
        x1 = np.arange(start, end, step)
        x2 = x1[::-1]
        #


from .utils import std_periodic_var


def compare_result(value1, value2, error1, error2=None, figname=None, yrange=None, periodic_vars=[]):
    diff_dict = {}
    if error2:
        for v in error1:
            if v in periodic_vars:
                diff = value1[v] - std_periodic_var(value2[v], value1[v])
            else:
                diff = value1[v] - value2[v]
            sigma = np.sqrt(error1[v] ** 2 + error2[v] ** 2)
            diff_dict[v] = diff / sigma
    else:
        for v in error1:
            if v in periodic_vars:
                diff = value1[v] - std_periodic_var(value2[v], value1[v])
            else:
                diff = value1[v] - value2[v]
            diff_dict[v] = diff / error1[v]
    if figname:
        arr = []
        if yrange:
            for v in diff_dict:
                if np.abs(diff_dict[v]) > yrange:
                    print("{0} out of yrange, which is {1}.".format(v, diff_dict[v]))
                    arr.append(np.sign(diff_dict[v]) * yrange)
                else:
                    arr.append(diff_dict[v])
            plt.ylim(-yrange, yrange)
        else:
            for v in diff_dict:
                arr.append(diff_dict[v])
        arr_x = np.arange(len(arr))
        plt.scatter(arr_x, arr)
        plt.xlabel("parameter index")
        plt.ylabel("sigma")
        plt.title(figname)
        plt.savefig(figname + ".png")
        plt.clf()
    return diff_dict
