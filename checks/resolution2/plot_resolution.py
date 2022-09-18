import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.data import batch_call, data_index, data_split
from tf_pwa.histogram import Hist1D


def sum_resolution(amps, weights, size=1):
    amps = tf.reshape(amps * weights, (-1, size))
    amps = tf.reduce_sum(amps, axis=-1).numpy()
    return amps


def main():
    config = ConfigLoader("config.yml")
    param = "final_params"
    if len(sys.argv) > 1:
        param = sys.argv[1]
    config.set_params(param + ".json")
    amp = config.get_amplitude()

    data = config.get_data("data_origin")[0]
    phsp = config.get_data("phsp_plot")[0]
    phsp_re = config.get_data("phsp_plot_re")[0]

    print("data loaded")
    amps = batch_call(amp, phsp_re, 20000)
    pw = batch_call(amp.partial_weight, phsp_re, 20000)

    re_weight = phsp_re["weight"]
    re_size = config.resolution_size
    print(sum_resolution(tf.ones_like(re_weight), re_weight, re_size))
    print(amps)
    amps = sum_resolution(amps, re_weight, re_size)
    print(np.argmax(amps), amps)

    pw = [sum_resolution(i, re_weight, re_size) for i in pw]

    m_idx = config.get_data_index("mass", "BC")
    m_phsp = data_index(phsp, m_idx).numpy()
    m_data = data_index(data, m_idx).numpy()

    m_min, m_max = np.min(m_phsp), np.max(m_phsp)

    scale = m_data.shape[0] / np.sum(amps)

    get_hist = lambda m, w: Hist1D.histogram(
        m, weights=w, range=(m_min, m_max), bins=120
    )

    data_hist = get_hist(m_data, None)
    phsp_hist = get_hist(m_phsp, scale * amps)
    pw_hist = []
    for i in pw:
        pw_hist.append(get_hist(m_phsp, scale * i))

    ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1)
    ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3, sharex=ax2)
    data_hist.draw_error(ax, label="data")
    phsp_hist.draw(ax, label="fit")

    for i, j in zip(pw_hist, config.get_decay()):
        i.draw_kde(
            ax, label=str(j.inner[-1])
        )  # "/".join([str(k) for k in j.inner]))

    (data_hist - phsp_hist).draw_pull(ax2)
    ax.set_ylim((0, None))
    ax.legend()
    ax.set_yscale("symlog")
    ax.set_ylabel("Events/{:.1f} MeV".format((m_max - m_min) * 10))
    ax2.set_xlabel("M( R_BC )")
    ax2.set_ylabel("pull")
    # ax2.set_xlim((1.3, 1.7))
    ax2.set_ylim((-5, 5))
    plt.setp(ax.get_xticklabels(), visible=False)
    print(param + "m_R_BC_fit.png")
    plt.savefig("m_R_BC_fit_" + param + ".png")


if __name__ == "__main__":
    main()
