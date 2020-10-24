#!/usr/bin/env python3
import sys
import os.path

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir + "/..")

from tf_pwa.config_loader import ConfigLoader

from tf_pwa.adaptive_bins import AdaptiveBound, cal_chi2
from tf_pwa.data import data_to_numpy, data_index
from tf_pwa.angle import kine_min, kine_max

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
import matplotlib.colors as mcolors


def cal_chi2_config(
    config, adapter, data, phsp, data_idx, bg=None, data_cut=None
):
    if data_cut is None:
        data_cut = np.array([data_index(data, idx) for idx in data_idx])
    amp_weight = config.get_amplitude()(phsp).numpy()
    phsp_cut = np.array([data_index(phsp, idx) for idx in data_idx])
    phsp_slice = np.concatenate(
        [np.array(phsp_cut ** 2), [amp_weight]], axis=0
    )
    phsps = adapter.split_data(phsp_slice)
    datas = adapter.split_data(data_cut ** 2)
    bound = adapter.get_bounds()
    if bg is not None:
        bg_weight = config._get_bg_weight(display=False)[0][0]
        bg_cut = np.array([data_index(bg, idx) for idx in data_idx])
        bgs = adapter.split_data(bg_cut ** 2)
        int_norm = (
            data_cut.shape[-1] - bg_cut.shape[-1] * bg_weight
        ) / np.sum(amp_weight)
    else:
        int_norm = data_cut.shape[-1] / np.sum(amp_weight)
    print("int norm:", int_norm)
    numbers = []
    for i, bnd in enumerate(bound):
        min_x, min_y = bnd[0]
        max_x, max_y = bnd[1]
        ndata = datas[i].shape[-1]
        nmc = np.sum(phsps[i][2]) * int_norm
        if bg is not None:
            nmc += bgs[i].shape[-1] * bg_weight
        numbers.append((ndata, nmc))
    return cal_chi2(numbers, config.get_ndf())


def draw_dalitz(data_cut, bound):
    fig, ax = plt.subplots()
    my_cmap = plt.get_cmap("jet")
    # my_cmap.set_under('w', 1)

    for i, bnd in enumerate(bound):
        min_x, min_y = bnd[0]
        max_x, max_y = bnd[1]
        rect = mpathes.Rectangle(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            linewidth=1,
            facecolor="none",
            edgecolor="black",
        )  # cmap(weights[i]/max_weight))
        ax.add_patch(rect)

    ah = ax.hist2d(
        data_cut[0] ** 2, data_cut[1] ** 2, bins=50, norm=mcolors.LogNorm()
    )
    # ax.scatter(data_cut[0]**2, data_cut[1]**2, s=1, c="red")
    ## using your own mass
    m0, mb, mc, md = 5.27926, 2.01026, 1.86961, 0.49368
    # print(ah)
    sbc_min, sbc_max = (mb + mc) ** 2, (m0 - md) ** 2
    sbd_min, sbd_max = (mb + md) ** 2, (m0 - mc) ** 2
    sbc = np.linspace(sbc_min, sbc_max, 1000)
    ax.plot(sbc, kine_max(sbc, m0, mc, mb, md), color="grey")
    ax.plot(sbc, kine_min(sbc, m0, mc, mb, md), color="grey")

    ax.set_xlim((sbc_min, sbc_max))
    ax.set_ylim((sbd_min, sbd_max))
    ax.set_xlabel("$M_{BC}$")
    ax.set_ylabel("$M_{BD}$")
    fig.colorbar(ah[-1])
    plt.savefig("figure/mbc_mbd_adaptive.png", dpi=200)


def main():
    config = ConfigLoader("config.yml")
    data = config.get_data("data")[0]
    phsp = config.get_data("phsp")[0]
    bg = config.get_data("bg")[0]
    config.set_params("final_params.json")

    mbc_idx = config.get_data_index(
        "mass", "R_BC"
    )  # ("particle", "(D, K)", "m")
    mbd_idx = config.get_data_index(
        "mass", "R_BD"
    )  # ("particle", "(D0, K, pi)", "m")
    data_idx = [mbc_idx, mbd_idx]

    data_cut = np.array([data_index(data, idx) for idx in data_idx])
    adapter = AdaptiveBound(data_cut ** 2, [[2, 2], [3, 3], [2, 2]])
    bound = adapter.get_bounds()

    cal_chi2_config(
        config, adapter, data, phsp, data_idx, bg=bg, data_cut=data_cut
    )
    draw_dalitz(data_cut, bound)


if __name__ == "__main__":
    main()
