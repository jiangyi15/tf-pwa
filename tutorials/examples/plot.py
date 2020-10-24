#!/usr/bin/env python3
import os
import datetime
import json
from math import pi
from scipy import interpolate
import matplotlib.pyplot as plt
from functools import reduce

import sys
import os.path

this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + "/..")

from fit_scipy_new import prepare_data, get_decay_chains, get_amplitude, load_params

import tf_pwa
from tf_pwa.amp import *
from tf_pwa.cal_angle import prepare_data_from_decay
from tf_pwa.utils import load_config_file
from tf_pwa.data import flatten_dict_data


def config_split(config):
    ret = {}
    for i in config:
        ret[i] = {i: config[i]}
    return ret


def part_config(config, name=None):
    name = name if name is not None else []
    if isinstance(name, str):
        print(name)
        return {name: config[name]}
    ret = {}
    for i in name:
        if i in config:
            ret[i] = config[i]
    return ret


def equal_pm(a, b):
    def remove_pm(s):
        ret = s
        if s.endswith("p"):
            ret = s[:-1]
        elif s.endswith("m"):
            ret = s[:-1]
        return ret

    return remove_pm(a) == remove_pm(b)


def part_combine_pm(config_list):
    ret = []
    for i in config_list:
        for j in ret:
            if equal_pm(i, j[0]):
                j.append(i)
                break
        else:
            ret.append([i])
    return ret


def hist_line(data, weights, bins, xrange=None, inter=1):
    y, x = np.histogram(data, bins=bins, range=xrange, weights=weights)
    x = (x[:-1] + x[1:]) / 2
    func = interpolate.interp1d(x, y, kind="quadratic")
    delta = (xrange[1] - xrange[0]) / bins / inter
    x_new = np.arange(x[0], x[-1], delta)
    y_new = func(x_new)
    return x_new, y_new


param_list_test = [
    "particle/A/m",
    "particle/B/m",
    "particle/C/m",
    "particle/D/m",
    "particle/(B, C)/m",
    "particle/(B, D)/m",
    "particle/(C, D)/m",
    "decay/A->(B, C)+D/(B, C)/ang/beta",
    "decay/(B, C)->B+C/B/ang/beta",
    "decay/A->(B, C)+D/(B, C)/ang/alpha",
    "decay/(B, C)->B+C/B/ang/alpha",
    "decay/A->(B, D)+C/(B, D)/ang/beta",
    "decay/(B, D)->B+D/B/ang/beta",
    "decay/A->(B, D)+C/(B, D)/ang/alpha",
    "decay/(B, D)->B+D/B/ang/alpha",
    "decay/A->(C, D)+B/(C, D)/ang/beta",
    "decay/(C, D)->D+C/D/ang/beta",
    "decay/A->(C, D)+B/(C, D)/ang/alpha",
    "decay/(C, D)->D+C/D/ang/alpha",
    "decay/(B, D)->B+D/B/aligned_angle/beta",
    "decay/(B, C)->B+C/B/aligned_angle/beta",
    "decay/(B, D)->B+D/D/aligned_angle/beta",
    "decay/(C, D)->D+C/D/aligned_angle/beta",
    "decay/(B, D)->B+D/B/aligned_angle/alpha",
    "decay/(B, D)->B+D/B/aligned_angle/gamma",
    "decay/(B, C)->B+C/B/aligned_angle/alpha",
    "decay/(B, C)->B+C/B/aligned_angle/gamma",
    "decay/(B, D)->B+D/D/aligned_angle/alpha",
    "decay/(B, D)->B+D/D/aligned_angle/gamma",
    "decay/(C, D)->D+C/D/aligned_angle/alpha",
    "decay/(C, D)->D+C/D/aligned_angle/gamma",
]

param_list = [
    "m_A",
    "m_B",
    "m_C",
    "m_D",
    "m_BC",
    "m_BD",
    "m_CD",
    "beta_BC",
    "beta_B_BC",
    "alpha_BC",
    "alpha_B_BC",
    "beta_BD",
    "beta_B_BD",
    "alpha_BD",
    "alpha_B_BD",
    "beta_CD",
    "beta_D_CD",
    "alpha_CD",
    "alpha_D_CD",
    "beta_BD_B",
    "beta_BC_B",
    "beta_BD_D",
    "beta_CD_D",
    "alpha_BD_B",
    "gamma_BD_B",
    "alpha_BC_B",
    "gamma_BC_B",
    "alpha_BD_D",
    "gamma_BD_D",
    "alpha_CD_D",
    "gamma_CD_D",
]

params_config = {
    "m_BC": {
        "xrange": (2.15, 2.65),
        "display": "$m_{ {D*}^{-}\pi^{+} }$",
        "bins": 50,
        "units": "Gev",
    },
    "m_BD": {
        "xrange": (4.0, 4.47),
        "display": "$m_{ {D*}^{-}{D*}^{0} }$",
        "bins": 47,
        "units": "GeV",
    },
    "m_CD": {
        "xrange": (2.15, 2.65),
        "display": "$m_{ {D*}^{0}\pi^{+} }$",
        "bins": 50,
        "units": "GeV",
    },
    "alpha_B_BD": {
        "xrange": (-pi, pi),
        "display": r"$\phi^{ {D*}^{-} }_{ {D*}^{0} {D*}^{-} }$",
        "bins": 50,
        "units": "",
        "legend": False,
    },
    "alpha_BD": {
        "xrange": (-pi, pi),
        "display": r"$ \phi_{ {D*}^{0} {D*}^{-} }$",
        "bins": 50,
        "units": "",
        "legend": False,
    },
    "cosbeta_BD": {
        "xrange": (-1, 1),
        "display": r"$\cos \theta_{ {D*}^{0} {D*}^{-} }$",
        "bins": 50,
        "units": "",
        "legend": False,
    },
    "cosbeta_B_BD": {
        "xrange": (-1, 1),
        "display": r"$\cos \theta^{ {D*}^{-} }_{ {D*}^{0} {D*}^{-} }$",
        "bins": 50,
        "units": "",
        "legend": False,
    },
    "alpha_B_BC": {
        "xrange": (-pi, pi),
        "display": r"$\phi^{ {D*}^{-} }_{ {D*}^{-}\pi^{+} }$",
        "bins": 50,
        "units": "",
        "legend": False,
    },
    "alpha_BC": {
        "xrange": (-pi, pi),
        "display": r"$ \phi_{ {D*}^{-} \pi^{+} }$",
        "bins": 50,
        "units": "",
        "legend": False,
    },
    "cosbeta_BC": {
        "xrange": (-1, 1),
        "display": r"$\cos \theta_{ {D*}^{-} \pi^{+} }$",
        "bins": 50,
        "units": "",
        "legend": False,
    },
    "cosbeta_B_BC": {
        "xrange": (-1, 1),
        "display": r"$\cos \theta^{ {D*}^{-} }_{ {D*}^{-}\pi^{+} }$",
        "bins": 50,
        "units": "",
        "legend": False,
    },
    "alpha_D_CD": {
        "xrange": (-pi, pi),
        "display": r"$\phi^{ {D*}^{0} }_{ {D*}^{0}\pi^{+} }$",
        "bins": 50,
        "units": "",
        "legend": False,
    },
    "alpha_CD": {
        "xrange": (-pi, pi),
        "display": r"$ \phi_{ {D*}^{0} \pi^{+} }$",
        "bins": 50,
        "units": "",
        "legend": False,
    },
    "cosbeta_CD": {
        "xrange": (-1, 1),
        "display": r"$\cos \theta_{ {D*}^{0} \pi^{+} }$",
        "bins": 50,
        "units": "",
        "legend": False,
    },
    "cosbeta_D_CD": {
        "xrange": (-1, 1),
        "display": r"$\cos \theta^{ {D*}^{0} }_{ {D*}^{0}\pi^{+} }$",
        "bins": 50,
        "units": "",
        "legend": False,
    },
}
params_list_map = dict(zip(param_list, param_list_test))


def map_idx_for_params():
    for i in range(len(param_list)):
        name = o_name = param_list[i]
        if name.startswith("beta"):
            name = "cos" + name
        if name in params_config:
            params_config[name]["idx"] = params_list_map[o_name]


map_idx_for_params()


def get_weight(amp, config_list, mcdata, norm_int=1.0, res_list=None, pm_combine=None):
    a_weight = {}
    if res_list is None:
        if pm_combine:
            res_list = part_combine_pm(config_list)
        else:
            res_list = [[i] for i in config_list]
    config_res = [part_config(config_list, i) for i in res_list]
    res_name = {}
    for i in range(len(res_list)):
        name = res_list[i]
        if isinstance(name, list):
            if len(name) > 1:
                name = reduce(lambda x, y: "{}+{}".format(x, y), res_list[i])
            else:
                name = name[0]
        res_name[i] = name
        amp.set_used_res(config_res[i])
        a_weight[i] = amp(mcdata, cached=True).numpy() * norm_int
    return a_weight, res_list, res_name


def get_idx_data(data, idx):
    data_idx = flatten_dict_data(data)
    return data_idx[idx].numpy()


def plot(params="final_params.json", res_list=None, pm_combine=True):
    dtype = "float64"
    w_bkg = 0.768331

    config_list = load_config_file("Resonances")

    decs, final_particles, decay = get_decay_chains(config_list)
    data, bg, mcdata = prepare_data(decs, particles=final_particles, dtype=dtype)
    amp = get_amplitude(decs, config_list, decay)
    load_params(amp, params)

    total = amp(mcdata)
    int_mc = tf.reduce_sum(total).numpy()
    n_data = data_shape(data)
    n_bg = data_shape(bg)
    norm_int = (n_data - w_bkg * n_bg) / int_mc

    a_weight, res_list, res_name = get_weight(
        amp, config_list, mcdata, norm_int, res_list, pm_combine
    )

    cmap = plt.get_cmap("jet")
    N = 3 + len(res_list)
    colors = [cmap(float(i) / N) for i in range(N)]
    colors = [
        "black",
        "red",
        "green",
        "blue",
        "yellow",
        "magenta",
        "cyan",
        "purple",
        "teal",
        "springgreen",
        "azure",
    ] + colors

    def plot_params(
        ax, name, bins=None, xrange=None, idx=0, display=None, units="GeV", legend=True
    ):
        fd = lambda x: x
        if name.startswith("cos"):
            fd = lambda x: np.cos(x)
        inter = 2
        color = iter(colors)
        if display is None:
            display = name
        data_hist = np.histogram(fd(get_idx_data(data, idx)), range=xrange, bins=bins)
        # ax.hist(fd(data[idx].numpy()),range=xrange,bins=bins,histtype="step",label="data",zorder=99,color="black")
        data_y, data_x = data_hist[0:2]
        data_x = (data_x[:-1] + data_x[1:]) / 2
        data_err = np.sqrt(data_y)
        ax.errorbar(
            data_x, data_y, yerr=data_err, fmt=".", color=next(color), zorder=-2
        )
        if bg is not None:
            ax.hist(
                fd(get_idx_data(bg, idx)),
                range=xrange,
                bins=bins,
                histtype="stepfilled",
                alpha=0.5,
                color="grey",
                weights=[w_bkg] * n_bg,
                label="bg",
                zorder=-1,
            )
            mc_bg = fd(np.append(get_idx_data(bg, idx), get_idx_data(mcdata, idx)))
            mc_bg_w = np.append([w_bkg] * n_bg, total.numpy() * norm_int)
        else:
            mc_bg = fd(get_idx_data(mcdata, idx))
            mc_bg_w = total.numpy() * norm_int
        x_mc, y_mc = hist_line(mc_bg, mc_bg_w, bins, xrange)
        # ax.plot(x_mc,y_mc,label="total fit")
        ax.hist(
            mc_bg,
            weights=mc_bg_w,
            bins=bins,
            range=xrange,
            histtype="step",
            color=next(color),
            label="total fit",
            zorder=100,
        )
        for i in a_weight:
            weights = a_weight[i]
            x, y = hist_line(
                fd(get_idx_data(mcdata, idx)), weights, bins, xrange, inter
            )
            y = y
            ax.plot(
                x,
                y,
                label=res_name[i],
                linestyle="solid",
                linewidth=1,
                color=next(color),
            )
        if legend:
            ax.legend(framealpha=0.5, fontsize="small")
        ax.set_ylabel("events/({:.3f} {})".format((x_mc[1] - x_mc[0]), units))
        ax.set_xlabel("{} {}".format(display, units))
        if xrange is not None:
            ax.set_xlim(xrange[0], xrange[1])
        ax.set_ylim(0, None)
        ax.set_title(display)

    plot_list = [
        "m_BC",
        "m_BD",
        "m_CD",
        "alpha_BD",
        "cosbeta_BD",
        "alpha_B_BD",
        "cosbeta_B_BD",
        "alpha_BC",
        "cosbeta_BC",
        "alpha_B_BC",
        "cosbeta_B_BC",
        "alpha_CD",
        "cosbeta_CD",
        "alpha_D_CD",
        "cosbeta_D_CD",
    ]
    n = len(plot_list)
    if not os.path.exists("figure"):
        os.mkdir("figure")
    # plt.style.use("classic")
    for i in range(n):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        name = plot_list[i]
        plot_params(ax, name, **params_config.get(name, {}))
        fig.savefig("figure/" + name + ".pdf")
        fig.savefig("figure/" + name + ".png", dpi=300)


def plot_fig(ax, a):
    pass


if __name__ == "__main__":
    test_res_list = [
        ["Zc_4025"],
        ["Zc_4160"],
        ["D1_2420", "D1_2420p"],
        ["D1_2430", "D1_2430p"],
        ["D2_2460", "D2_2460p"],
    ]
    with tf.device("/device:CPU:0"):
        plot("final_params.json", res_list=None)
