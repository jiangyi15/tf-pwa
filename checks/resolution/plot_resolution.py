import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.data import data_index, data_split
from tf_pwa.histogram import Hist1D


def main():
    config = ConfigLoader("config.yml")
    config.set_params("final_params.json")
    amp = config.get_amplitude()

    data = config.get_data("data_origin")[0]
    phsp = config.get_data("phsp_plot")[0]
    phsp_re = config.get_data("phsp_plot_re")[0]

    print("data loaded")
    amps = amp(phsp_re)

    amps = tf.reshape(amps * phsp_re["weight"], (-1, config.resolution_size))
    amps = tf.reduce_sum(amps, axis=-1).numpy()

    m_idx = config.get_data_index("mass", "R_BC")
    m_phsp = data_index(phsp, m_idx).numpy()
    m_data = data_index(data, m_idx).numpy()

    m_min, m_max = np.min(m_phsp), np.max(m_phsp)

    scale = m_data.shape[0] / np.sum(amps)

    get_hist = lambda m, w: Hist1D.histogram(
        m, weights=w, range=(m_min, m_max), bins=200
    )

    data_hist = get_hist(m_data, None)
    phsp_hist = get_hist(m_phsp, scale * amps)
    data_hist.draw_error()
    phsp_hist.draw()
    plt.ylim((0, None))
    plt.xlabel("mass R_BC")
    plt.xlim((1.3, 1.7))
    plt.savefig("re_plot.png")


if __name__ == "__main__":
    main()
