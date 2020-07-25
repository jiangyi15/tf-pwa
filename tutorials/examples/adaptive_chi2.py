#!/usr/bin/env python3
import sys
import os.path

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir + '/..')

from tf_pwa.config_loader import ConfigLoader

from tf_pwa.adaptive_bins import AdaptiveBound
from tf_pwa.data import data_to_numpy, data_index
from tf_pwa.angle import kine_min, kine_max

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
import matplotlib.colors as mcolors


def cal_chi2(config, adapter, data, phsp, data_idx, bg=None, data_cut=None):
    if data_cut is None:
        data_cut = np.array([data_index(data, idx) for idx in data_idx])
    amp_weight = config.get_amplitude()(phsp).numpy()
    phsp_cut = np.array([data_index(phsp, idx) for idx in data_idx])
    phsp_slice = np.concatenate([np.array(phsp_cut**2), [amp_weight]], axis=0)
    phsps = adapter.split_data(phsp_slice)
    datas = adapter.split_data(data_cut**2)
    bound = adapter.get_bounds()
    if bg is not None:
        bg_weight = config.get_bg_weight(display=False)[0]
        bg_cut = np.array([data_index(bg, idx) for idx in data_idx])
        bgs = adapter.split_data(bg_cut**2)
        int_norm = (data_cut.shape[-1] - bg_cut.shape[-1] * bg_weight)/ np.sum(amp_weight)
    else:
        int_norm = data_cut.shape[-1] / np.sum(amp_weight)
    print("int norm:", int_norm)
    weights = []
    chi21 = []
    for i, bnd in enumerate(bound):
        min_x, min_y = bnd[0]
        max_x, max_y = bnd[1]
        ndata = datas[i].shape[-1]
        nmc = np.sum(phsps[i][2]) * int_norm
        if bg is not None:
            nmc += bgs[i].shape[-1] * bg_weight
        weight = (ndata - nmc) / np.sqrt(ndata)
        weights.append(weight**2)
        chi21.append(ndata * np.log(nmc))
    max_weight = np.max(weights)
    chi2 = np.sum(weights)
    n_fp = config.get_ndf()
    print("bins: ", len(bound))
    print("number of free parameters: ", n_fp)
    ndf = len(bound) - 1 - n_fp
    print("chi2/ndf: ", np.sum(weights), "/", ndf) # ,"another", np.sum(chi21))
    return chi2, ndf


def draw_dalitz(data_cut, bound):
    fig, ax = plt.subplots()
    my_cmap = plt.get_cmap("jet")
    # my_cmap.set_under('w', 1)

    
    for i, bnd in enumerate(bound):
        min_x, min_y = bnd[0]
        max_x, max_y = bnd[1]
        rect = mpathes.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, linewidth=1, facecolor="none", edgecolor="black") #cmap(weights[i]/max_weight))
        ax.add_patch(rect)
    
    ah =  ax.hist2d(data_cut[0]**2, data_cut[1]**2, bins=50, norm=mcolors.LogNorm())
    m0, m1, m2, m3 = 5.27926, 2.01026, 1.86961, 0.49368
    # print(ah)
    s12_min, s12_max = (m1 + m2)**2, (m0 - m3)**2
    s13_min, s13_max = (m1 + m3)**2, (m0 - m2)**2
    s12 = np.linspace(s12_min, s12_max, 1000)
    ax.plot(s12, kine_max(s12, m0, m2, m1, m3), color="grey")
    ax.plot(s12, kine_min(s12, m0, m2, m1, m3), color="grey")
    
    ax.set_xlim((s12_min, s12_max))
    ax.set_ylim((s13_min, s13_max))
    ax.set_xlabel("$M_{12}$")
    ax.set_ylabel("$M_{13}$")
    fig.colorbar(ah[-1])
    plt.savefig("figure/m_12_m13_adaptive.png", dpi=200)


def main():
    config = ConfigLoader("config.yml")
    data = config.get_data("data")
    phsp = config.get_data("phsp")
    bg = config.get_data("bg")
    config.set_params("final_params.json")
    
    m12_idx = ("particle", "(D, D0, pi)", "m")
    m13_idx = ("particle", "(D0, K, pi)", "m")
    data_idx = [m12_idx, m13_idx]

    data_cut = np.array([data_index(data, idx) for idx in data_idx])
    adapter = AdaptiveBound(data_cut**2, [[2,2]]*4)
    bound = adapter.get_bounds()
    
    cal_chi2(config, adapter, data, phsp, data_idx, bg=bg, data_cut=data_cut)
    draw_dalitz(data_cut, bound)


if __name__ == "__main__":
    main()
