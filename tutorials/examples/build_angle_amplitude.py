"""
scripts for calcaulate split angle amplitude
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.data import data_index, data_to_numpy
from tf_pwa.experimental.build_amp import build_angle_amp_matrix


def main():
    config = ConfigLoader("config.yml")
    decay = config.get_amplitude().decay_group
    data = config.get_data("phsp")
    ret = []
    for i in data:
        cached_amp = build_angle_amp_matrix(decay, i)
        ret.append(data_to_numpy(cached_amp))

    idx = config.get_data_index("angle", "R_BC/B")
    ang = data_index(data[0], idx)

    np.savez("phsp.npz", ret)

    for k, v in ret[0].items():
        for i, amp in enumerate(v):
            w = np.abs(amp) ** 2
            w = np.sum(np.reshape(w, (amp.shape[0], -1)), axis=-1)
            plt.hist(
                np.cos(ang["beta"]),
                weights=w,
                bins=20,
                histtype="step",
                label="{}: {}".format(k, i),
            )
    plt.savefig("angle_costheta.png")


if __name__ == "__main__":
    main()
