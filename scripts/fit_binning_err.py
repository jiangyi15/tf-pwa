import operator
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.data import data_index, data_mask
from tf_pwa.histogram import Hist1D
from tf_pwa.model.model import sum_gradient


def binning_gradient(binning, amp, phsp, m_phsp, var):
    """
    calculate bin error for fit
    including error propagation (gradients) and weights
    """
    nfit = []
    for i in range(binning.shape[0] - 1):
        m_min = binning[i]
        m_max = binning[i + 1]
        phsp_i = data_mask(phsp, (m_phsp > m_min) & (m_phsp < m_max))
        weight_i = phsp_i.get("weight", 1.0)
        with tf.GradientTape() as tape:
            amp_i = amp(phsp_i)
            int_mc = tf.reduce_sum(weight_i * amp_i)
        grad = tape.gradient(int_mc, var, unconnected_gradients="zero")
        w_error2 = tf.reduce_sum(weight_i ** 2 * amp_i ** 2)
        grad = np.array([j.numpy() for j in grad])
        nfit.append((int_mc.numpy(), grad, w_error2.numpy()))
    count_i = np.array([i[0] for i in nfit])
    sum_int = np.sum(count_i)
    grads = np.array([i[1] for i in nfit])
    sum_int_grad = np.sum(grads, axis=0)
    w_error2 = np.array([i[2] for i in nfit]) / sum_int / sum_int
    grads = (grads - sum_int_grad * count_i[:, None] / sum_int) / sum_int
    return count_i / sum_int, grads, w_error2


def main():
    Nbins = 72
    config = ConfigLoader("config.yml")
    config.set_params("final_params.json")
    error_matrix = np.load("error_matrix.npy")

    idx = config.get_data_index("mass", "R_BC")

    datas = config.get_data("data")
    phsps = config.get_data("phsp")
    bgs = config.get_data("bg")
    amp = config.get_amplitude()
    var = amp.trainable_variables

    get_data = lambda x: data_index(x, idx).numpy()

    m_all = np.concatenate([get_data(i) for i in datas])

    m_min = np.min(m_all) - 0.1
    m_max = np.max(m_all) + 0.1
    binning = np.linspace(m_min, m_max, Nbins + 1)

    get_hist = lambda x, w: Hist1D.histogram(
        get_data(x), bins=Nbins, range=(m_min, m_max), weights=w
    )

    data_hist = [get_hist(i, i.get("weight")) for i in datas]
    bg_hist = [get_hist(i, np.abs(i.get("weight"))) for i in bgs]

    phsp_hist = []
    for dh, bh, phsp in zip(data_hist, bg_hist, phsps):
        m_phsp = data_index(phsp, idx).numpy()

        y_frac, grads, w_error2 = binning_gradient(
            binning, amp, phsp, m_phsp, var
        )
        error2 = np.einsum("ij,jk,ik->i", grads, error_matrix, grads)
        # error parameters and error from integration sample weights
        yerr = np.sqrt(error2 + w_error2)

        n_fit = dh.get_count() - bh.get_count()
        phsp_hist.append(n_fit * Hist1D(binning, y_frac, yerr))

    total_data = reduce(operator.add, data_hist)
    ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    total_data.draw(ax, label="data")
    total_data.draw_error(ax)
    total_bg = reduce(operator.add, bg_hist)
    total_bg.draw_bar(ax, label="back ground", color="grey", alpha=0.5)
    total_fit = reduce(operator.add, phsp_hist + bg_hist)
    total_fit.draw(ax, label="fit")
    total_fit.draw_error(ax)
    ax.set_ylim((0, None))
    ax.set_xlim((m_min, m_max))
    ax.legend()
    ax.set_ylabel(f"Events/ {(m_max-m_min)/Nbins:.3f} GeV")
    ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1)
    (total_data - total_fit).draw_pull(ax2)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax2.set_ylim((-5, 5))
    ax2.set_xlim((m_min, m_max))
    ax2.axhline(0, c="black", ls="-")
    ax2.axhline(-3, c="r", ls="--")
    ax2.axhline(3, c="r", ls="--")
    ax2.set_xlabel("$M(BC)$/GeV", loc="right")
    plt.savefig("fit_full_error.png")


if __name__ == "__main__":
    with tf.device("CPU"):
        main()
