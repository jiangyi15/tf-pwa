import operator
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tf_pwa.config_loader import ConfigLoader, MultiConfig
from tf_pwa.data import data_index, data_mask
from tf_pwa.histogram import Hist1D, WeightedData


def sum_hist(x):
    return reduce(operator.add, x)


def plot_mass(
    amp, datas, bgs, phsps, get_data, name, Nbins, xrange=None, unit="GeV"
):

    # determine histogram range
    if xrange is None:
        m_all = np.concatenate([get_data(i) for i in datas])
        m_min, m_max = np.min(m_all), np.max(m_all)
        delta_m = (m_max - m_min) / Nbins
        m_min, m_max = m_min - 3 * delta_m, m_max + 3 * delta_m
    else:
        m_min, m_max = xrange

    binning = np.linspace(m_min, m_max, Nbins + 1)

    get_hist = lambda x, w: Hist1D.histogram(
        get_data(x), bins=Nbins, range=(m_min, m_max), weights=w
    )
    # using more bins to smooth
    get_hist2 = lambda x, w: WeightedData(
        get_data(x), bins=Nbins * 2, range=(m_min, m_max), weights=w * 2
    )
    # get data and background histogram
    data_hist = [get_hist(i, i.get("weight")) for i in datas]
    bg_hist = [get_hist(i, np.abs(i.get("weight"))) for i in bgs]
    phsp_hist = []
    pw_hist = []
    for dh, bh, phsp in zip(data_hist, bg_hist, phsps):
        # using amplitude square as weights for fit
        amp_i = phsp.get("weight", 1.0) * amp(phsp)
        int_mc = tf.reduce_sum(amp_i)
        # normalize to signal number
        scale = (dh.get_count() - bh.get_count()) / int_mc
        y_frac = amp_i * scale
        phsp_hist.append(get_hist(phsp, y_frac.numpy()))

        # calculate partial wave weights
        pw = amp.partial_weight(phsp)
        tmp = []
        for i in pw:
            y_frac = phsp.get("weight", 1.0) * i * scale
            tmp.append(get_hist2(phsp, y_frac.numpy()))
        pw_hist.append(tmp)

    total_data = sum_hist(data_hist)
    total_bg = sum_hist(bg_hist)
    total_fit = sum_hist(phsp_hist + bg_hist)

    # plot data and fit histogram
    ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    total_data.draw(ax, label="data")
    total_data.draw_error(ax)
    total_bg.draw_bar(ax, label="back ground", color="grey", alpha=0.5)
    total_fit.draw(ax, label="fit")
    # total_fit.draw_error(ax)
    for hi, dec in zip(zip(*pw_hist), amp.decay_group):
        h = sum_hist(hi)
        res = dec[0].outs
        name_i = " ".join([getattr(i, "display", str(i)) for i in res])
        # h.draw(ax, label=name_i)
        h.draw_kde(ax, label=name_i)
    # plot legend
    # ax.legend()
    # plot pull
    ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1)
    (total_data - total_fit).draw_pull(ax2)
    ax2.axhline(0, c="black", ls="-")
    ax2.axhline(-3, c="r", ls="--")
    ax2.axhline(3, c="r", ls="--")
    ax.grid()
    ax2.grid()
    # set some axis attributes
    ax.set_ylim((0, None))
    ax.set_xlim((m_min, m_max))
    ax.set_ylabel(f"Events/ {(m_max-m_min)/Nbins:.3f} {unit}")
    plt.setp(ax.get_xticklabels(), visible=False)
    ax2.set_ylim((-5, 5))
    ax2.set_xlim((m_min, m_max))
    ax2.set_xlabel(f"${name}$/{unit}", loc="right")
    # save figure
    plt.savefig(f"{name}.png", dpi=300)


def main():
    Nbins = 64
    config = ConfigLoader("config.yml")
    # config = MultiConfig(["config.yml"]).configs[0]
    config.set_params("final_params.json")
    name = "R_BC"
    idx = config.get_data_index("mass", "R_BC")
    # idx_costheta = (*config.get_data_index("angle", "DstD/D*"), "beta")

    datas, phsps, bgs, _ = config.get_all_data()
    amp = config.get_amplitude()
    get_data = lambda x: data_index(x, idx).numpy()
    # get_data = lambda x: np.cos(data_index(x, idx_costheta).numpy())
    plot_mass(amp, datas, bgs, phsps, get_data, name, Nbins)


if __name__ == "__main__":
    main()
