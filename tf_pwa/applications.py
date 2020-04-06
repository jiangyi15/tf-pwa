"""
This module provides functions that implements user-friendly interface to the functions and methods in other modules.
It acts like a synthesis of all the other modules of their own physical purposes.
In general, users only need to import functions in this module to implement their physical analysis instead of
going into every modules. There are some example files where you can figure out how it is used.
"""
import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm as Norm
from iminuit import Minuit

from .fitfractions import cal_fitfractions, cal_fitfractions_no_grad
from .data import split_generator, load_dat_file, data_to_tensor
from .cal_angle import prepare_data_from_decay, cal_angle_from_momentum
from .phasespace import PhaseSpaceGenerator
from .fit import fit_scipy, fit_minuit, fit_multinest
from .significance import significance
from .utils import error_print, std_periodic_var


def fit_fractions(model, mcdata, inv_he, hesse=False):
    """
    This function calculate fit fractions of the resonances as well as their coherent pairs. It imports
    ``cal_fitfractions`` and ``cal_fitfractions_no_grad`` from module **tf_pwa.fitfractions**.

    .. math:: FF_{i} = \\frac{\\int |A_i|^2 d\\Omega }{ \\int |\\sum_{i}A_i|^2 d\\Omega }\\approx \\frac{\\sum |A_i|^2 }{\\sum|\\sum_{i} A_{i}|^2}

    gradients???:

    .. math:: FF_{i,j} = \\frac{\\int 2Re(A_i A_j*) d\\Omega }{ \\int |\\sum_{i}A_i|^2 d\\Omega } = \\frac{\\int |A_i +A_j|^2  d\\Omega }{ \\int |\\sum_{i}A_i|^2 d\\Omega } - FF_{i} - FF_{j}

    hessians:

    .. math:: \\frac{\\partial }{\\partial \\theta_i }\\frac{f(\\theta_i)}{g(\\theta_i)} = \\frac{\\partial f(\\theta_i)}{\\partial \\theta_i} \\frac{1}{g(\\theta_i)} - \\frac{\\partial g(\\theta_i)}{\\partial \\theta_i} \\frac{f(\\theta_i)}{g^2(\\theta_i)}

    :param model: Model object.
    :param mcdata: MCdata array.
    :param inv_he: The inverse of Hessian matrix.
    :param hesse: Boolean. Whether to calculate the error as well.
    :return frac: Dictionary of fit fractions for each resonance.
    :return err_frac: Dictionary of their errors. If ``hesse`` is ``False``, it will be a dictionary of ``None``.
    """
    err_frac = {}
    if hesse:
        frac, grad = cal_fitfractions(model.Amp, list(split_generator(mcdata, 25000)))
        for i in frac:
            err_frac[i] = np.sqrt(np.dot(np.dot(inv_he, grad[i]), grad[i]))
    else:
        frac = cal_fitfractions_no_grad(model.Amp, list(split_generator(mcdata, 45000)))
        for i in frac:
            err_frac[i] = None
    return frac, err_frac


def corr_coef_matrix(npy_name):
    """
    This function obtains correlation coefficients matrix of all trainable variables from *.npy file.

    :param npy_name: String. Name of the npy file
    :return: Numpy 2-d array. The correlation coefficient matrix.
    """
    err_mtx = np.load(npy_name)
    err = np.sqrt(err_mtx.diagonal())
    diag_mtx = np.diag(1 / err)
    tmp_mtx = np.matmul(diag_mtx, err_mtx)
    cc_mtx = np.matmul(tmp_mtx, diag_mtx)
    return cc_mtx


'''
def calPWratio(params, POLAR=True):
    """
    This function calculates the ratio of different partial waves in a certain resonance using the input values
    of fitting parameters. It is useful when user check if one partial wave is too small compared to the other partial
    waves, in which case, the fitting result may be not reliable since it has potential to give nuisance
    degree of freedom. (WIP)

    :param params: Dictionary of values indexed by the name of the fitting parameters
    :param POLAR: Boolean. Whether the parameters are defined in the polar coordinate or the Cartesian coordinate.
    :return: None. But the function will print the ratios.
    """
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
'''


def cal_hesse_error(model):
    """
    This function calculates the errors of all trainable variables.
    The errors are given by the square root of the diagonal of the inverse Hessian matrix.

    :param model: Model.
    :return hesse_error: List of errors.
    :return inv_he: The inverse Hessian matrix.
    """
    t = time.time()
    nll, g, h = model.cal_nll_hessian()  # data_w,mcdata,weight=weights,batch=50000)
    print("Time for calculating errors:", time.time() - t)
    inv_he = np.linalg.pinv(h.numpy())
    np.save("error_matrix.npy", inv_he)
    diag_he = inv_he.diagonal()
    hesse_error = np.sqrt(np.fabs(diag_he)).tolist()
    return hesse_error, inv_he


def gen_data(amp, particles, Ndata, mcfile, Nbg=0, wbg=0, Poisson_fluc=False,
             bgfile=None, genfile=None):
    """
    This function is used to generate toy data according to an amplitude model.

    :param amp: AmplitudeModel???
    :param particles: List of final particles
    :param Ndata: Integer. Number of data
    :param mcfile: String. The MC sample file used to generate signal data.
    :param Nbg: Integer. Number of background. By default it's 0.
    :param wbg: Float. Weight of background. By default it's 0.
    :param Poisson_fluc: Boolean. If it's ``True``, The number of data will be decided by a Poisson distribution around the given value.
    :param bgfile: String. The background sample file used to generate a certain number of background data.
    :param genfile: String. The file to store the generated toy.
    :return: tensorflow.Tensor. The generated toy data.
    """
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

    if isinstance(genfile, str):
        np.savetxt(genfile, data_gen)
    # data = prepare_data_from_decay(genfile, amp.decay_group, particles=particles, dtype=dtype)

    momenta = {}
    Npar = len(particles)
    for p in range(Npar):
        momenta[particles[p]] = data_gen[p::Npar]
    data = cal_angle_from_momentum(momenta, amp.decay_group)
    return data_to_tensor(data)


def gen_mc(mother, daughters, number, outfile="data/flat_mc.dat"):
    """
    This function generates phase-space MC data (without considering the effect of detector performance). It imports
    ``PhaseSpaceGenerator`` from module **tf_pwa.phasespace**.

    :param mother: Float. The invariant mass of the mother particle.
    :param daughters: List of float. The invariant masses of the daughter particles.
    :param number: Integer. The number of MC data generated.
    :param outfile: String. The file to store the generated MC.
    :return: Numpy array. The generated MC data.
    """
    # 4.59925172, [2.00698, 2.01028, 0.13957] DBC: D*0 D*- pi+
    phsp = PhaseSpaceGenerator(mother, daughters)
    flat_mc_data = phsp.generate(number)
    pf = []
    for i in range(len(daughters)):
        pf.append(flat_mc_data[i])
    pf = np.transpose(pf, (1, 0, 2)).reshape((-1, 4))
    np.savetxt(outfile, pf)  # 一个不包含探测器效率的MC样本
    return pf


def fit(Use="scipy", **kwargs):
    """
    Fit the amplitude model using ``scipy``, ``iminuit`` or ``pymultinest``. It imports
    ``fit_scipy``, ``fit_minuit``, ``fit_multinest`` from module **tf_pwa.fit**.

    :param Use: String. If it's ``"scipy"``, it will call ``fit_scipy``; if it's ``"minuit"``, it will call ``fit_minuit``; if it's ``"multinest"``, it will call ``fit_multinest``.
    :param kwargs: The arguments will be passed to the three functions above.

    For ``fit_scipy``

    :param fcn: FCN object to be minimized.
    :param method: String. Options in ``scipy.optimize``. For now, it implements interface to such as "BFGS", "L-BFGS-B", "basinhopping".
    :param bounds_dict: Dictionary of boundary constrain to variables.
    :param kwargs: Other arguments passed on to ``scipy.optimize`` functions.
    :return: FitResult object, List of NLLs, List of point arrays.

    For ``fit_minuit``

    :param fcn: FCN object to be minimized.
    :param bounds_dict: Dictionary of boundary constrain to variables.
    :param hesse: Boolean. Whether to call ``hesse()`` after ``migrad()``. It's ``True`` by default.
    :param minos: Boolean. Whether to call ``minos()`` after ``hesse()``. It's ``False`` by default.
    :return: Minuit object

    For ``fit_multinest`` (WIP)

    :param fcn: FCN object to be minimized.
    """
    if Use == "scipy":
        ret = fit_scipy(**kwargs)
    elif Use == "minuit":
        ret = fit_minuit(**kwargs)
    # elif Use == "multinest":
    #    ret = fit_multinest(**kwargs)
    else:
        raise Exception("Unknown fit tool {}".format(Use))
    return ret


def cal_significance(nll1, nll2, ndf):
    """
    This function calculates the statistical significance.

    :param nll1: Float. NLL of the first PDF.
    :param nll2:  Float. NLL of the second PDF.
    :param ndf: The difference of the degrees of freedom of the two PDF.
    :return: Float. The statistical significance
    """
    sigma = significance(nll1, nll2, ndf)
    return sigma


def plot_pull(data, name, nbins=20, norm=False, value=None, error=None):
    """
    This function is used to plot the pull for a data sample.

    :param data: List
    :param name: String. Name of the sample
    :param nbins: Integer. Number of bins in the histogram
    :param norm: Boolean. Whether to normalize the histogram
    :param value: Float. Mean value in normalization
    :param error: Float or list. Sigma value(s) in normalization
    :return: The fitted mu, sigma, as well as their errors
    """
    data = np.array(data)
    if norm:
        if value is None or error is None:
            raise Exception("Need value or error for normed pull!")
        data = (data - value) / error

    _, bins, _ = plt.hist(data, nbins, density=True, alpha=0.6)
    bins = np.linspace(bins[0], bins[-1], 30)

    def fitNormHist(data):
        def nll(mu, sigma):
            def normpdf(x, mu, sigma):
                return np.exp(-(x - mu) * (x - mu) / 2 / sigma / sigma) / np.sqrt(2 * np.pi) / sigma

            return -np.sum(np.log(normpdf(data, mu, sigma)))

        m = Minuit(nll, mu=0, sigma=1, error_mu=0.1, error_sigma=0.1, errordef=0.5)
        m.migrad()
        m.hesse()
        return m

    m = fitNormHist(data)
    mu, sigma = m.values["mu"], m.values["sigma"]
    err_mu, err_sigma = m.errors["mu"], m.errors["sigma"]
    y = Norm.pdf(bins, mu, sigma)
    plt.plot(bins, y, "r-")
    plt.xlabel(name)
    plt.title("mu = " + error_print(mu, err_mu) + '; sigma = ' + error_print(sigma, err_sigma))
    plt.savefig("fig/" + name + "_pull.png")
    plt.clf()
    return mu, sigma, err_mu, err_sigma


# def likelihood_profile(var_name, start=None, end=None, step=None, values=None, errors=None, mode="bothsides"):
# if start == None or end == None:
#    x_mean = values[var_name]
#    x_sigma = errors[var_name]
#    start = x_mean - 10 * x_sigma
#    end = x_mean + 10 * x_sigma
# else:
#    x_mean = (end + start) / 2
# if step == None:
#    step = (end - start) / 100
# if mode == "bothsides":
#    x1 = np.arange(x_mean, start, -step)
#    x2 = np.arange(x_mean, end, step)
# elif mode == "back&forth":
#    x1 = np.arange(start, end, step)
#    x2 = x1[::-1]

def likelihood_profile(m, var_names, bins=20, minos=True):
    """
    Calculate the likelihood profile for a variable.

    :param m: Minuit object
    :param var_names: Either a string or a list of strings
    :param bins: Integer
    :param minos: Boolean. If it's ``False``, the function will call ``Minuit.profile()`` to derive the 1-d scan of **var_names**; if it's ``True``, the function will call ``Minuit.mnprofile()`` to derive the likelihood profile, which is much more time-consuming.
    :return: Dictionary indexed by **var_names**. It contains the return of either ``Minuit.mnprofile()`` or ``Minuit.profile()``.
    """
    if isinstance(var_names, str):
        var_names = [var_names]
    lklpf = {}
    for var in var_names:
        if minos:
            x, y, t = m.mnprofile(var, bins=bins)
            lklpf[var] = [x, y, t]
        else:
            x, y = m.profile(var, bins=bins)
            lklpf[var] = [x, y]
    return lklpf


def compare_result(value1, value2, error1, error2=None, figname=None, yrange=None, periodic_vars=None):
    """
    Compare two groups of fitting results. If only one error is provided,
    the figure is :math:`\\frac{\\mu_1-\\mu_2}{\\sigma_1}`;
    if both errors are provided, the figure is :math:`\\frac{\\mu_1-\\mu_2}{\\sqrt{\\sigma_1^2+\\sigma_2^2}}`.

    :param value1: Dictionary
    :param value2: Dictionary
    :param error1: Dictionary
    :param error2: Dictionary. By default it's ``None``.
    :param figname: String. The output file
    :param yrange: Float. If it's not given, there is no y-axis limit in the figure.
    :param periodic_vars: List of strings. The periodic variables.
    :return: Dictionary of quality figure of each variable.
    """
    if periodic_vars is None:
        periodic_vars = []
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
