import matplotlib.pyplot as plt
import numpy as np
from detector import (
    delta_function,
    detector_config,
    phsp_factor,
    rec_function,
    truth_function,
)

from tf_pwa.config_loader import ConfigLoader

if __name__ == "__main__":
    config = ConfigLoader("config.yml")

    phsp_truth = config.data.load_data("data/phsp_truth.dat")
    phsp_rec = config.data.load_data("data/phsp_rec.dat")

    m1 = phsp_truth.get_mass("(B, C)").numpy()
    m2 = phsp_rec.get_mass("(B, C)").numpy()

    mi = np.linspace(0.2, 1.9, 1000)
    plt.hist(m2, bins=400, range=[0.0, 2.0], alpha=0.5, label="rec")
    plt.hist(m1, bins=400, range=[0.0, 2.0], alpha=0.5, label="truth")
    f1 = truth_function(mi)
    n1 = np.mean(f1) * (1.9 - 0.2) / (m1.shape[0] * 2.0 / 400)
    f2 = rec_function(mi)
    n2 = np.mean(f2) * (1.9 - 0.2) / (m1.shape[0] * 2.0 / 400)
    plt.plot(mi, f2 / n2, label="rec function")
    plt.plot(mi, f1 / n1, label="truth function")
    plt.legend()
    plt.savefig("m_rec.png")

    plt.clf()

    sigma = detector_config["sigma"]
    bias = detector_config["bias"]
    min_delta = -sigma * 5 + bias
    max_delta = sigma * 5 + bias
    plt.hist(
        m2 - m1, bins=1000, range=[min_delta, max_delta]
    )  # [-0.15, 0.15])
    delta = np.linspace(min_delta, max_delta, 1000)
    f3 = np.exp(-((delta - bias) ** 2) / 2 / sigma**2)
    n3 = np.mean(f3) / (m1.shape[0] / 1000)
    plt.plot(delta, f3 / n3, label="gauss function")
    f4 = delta_function(delta)
    n4 = np.mean(f4) / (m1.shape[0] / 1000)
    plt.plot(delta, f4 / n4, label="$f(\\delta)$")
    plt.xlabel("$\delta$ mass")
    plt.legend()
    plt.xlim((min_delta, max_delta))
    plt.xlabel("M(rec)-M(truth)")
    plt.savefig("m_diff.png")

    plt.clf()
    mi = np.linspace(0.2 + 1e-10, 1.9 - 1e-10, 1000)
    fe = truth_function(mi) / phsp_factor(mi)
    plt.plot(
        mi,
        fe * 1.7 / np.sqrt(2 * np.pi) / detector_config["sigma"],
        label="eff with cut",
    )
    plt.plot(mi, mi, label="y=x")
    plt.legend()
    plt.ylim((0, None))
    plt.savefig("m_eff.png")
