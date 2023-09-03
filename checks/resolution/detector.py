"""
Example of detector model

"""

import numpy as np
import tensorflow as tf
import yaml

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.data import data_mask
from tf_pwa.data_trans.helicity_angle import HelicityAngle

config = ConfigLoader("config.yml")
decay_chain = config.get_decay(False).get_decay_chain("BC")
ha = HelicityAngle(decay_chain)

# loda detector model parameters
with open("detector.yml") as f:
    detector_config = yaml.safe_load(f)


def detector(data):
    """
    run detector for data
    """

    ms, costheta, phi = ha.find_variable(data)
    p_map = {str(i): i for i in ms.keys()}
    m = ms[p_map["BC"]]

    # smear mass
    new_m = (
        m
        + tf.random.normal(shape=m.shape, dtype=m.dtype)
        * detector_config["sigma"]
        + detector_config["bias"]
    )

    # selected with effecency, eff(m) = m
    weight = m
    max_w = 2.0
    cut = max_w * tf.random.uniform(shape=m.shape, dtype=m.dtype) < weight
    # selected in reconstructed value, a mass cut
    cut = (cut) & (new_m < 1.9) & (new_m > 0.2)

    toy_truth = data_mask(data, cut)
    ms[p_map["BC"]] = new_m
    toy_rec = ha.build_data(*data_mask((ms, costheta, phi), cut))
    toy_rec = config.data.cal_angle(toy_rec)

    return toy_truth, toy_rec


def relative_p(m0, m1, m2):
    """breakup monmentum"""
    s2 = (m0**2 - (m1 + m2) ** 2) * (m0**2 - (m1 - m2) ** 2)
    s2 = np.where(s2 < 0, 0, s2)
    return np.sqrt(s2) / 2 / m0


def phsp_factor(m):
    """phase space factor"""
    p1 = relative_p(2.0, m, 0.1)
    p2 = relative_p(m, 0.1, 0.1)
    return p1 * p2


def gauss(delta, sigma):
    """simple gauss function without normalisation"""
    return np.exp(-(delta**2) / 2 / sigma**2)


def log_gauss(delta, sigma):
    """simple gauss function without normalisation"""
    return -(delta**2) / 2 / sigma**2


def trans_function(m1, m2):
    """
    transisation function

    """

    delta = m2 - m1 - detector_config["bias"]
    r1 = gauss(delta, sigma=detector_config["sigma"])
    eff = np.where((m2 > 0.2) & (m2 < 1.9), m1, 0.0) * phsp_factor(m1)
    return r1 * eff


def log_trans_function(m1, m2):
    """
    log transisation function

    """

    delta = m2 - m1 - detector_config["bias"]
    r1 = log_gauss(delta, sigma=detector_config["sigma"])
    eff = np.where((m2 > 0.2) & (m2 < 1.9), m1, 0.0) * phsp_factor(m1)
    cut = np.where(eff < 1e-6, 0.0, 1.0)
    return r1 + np.log(np.where(cut == 0, 1, eff)), cut


def rec_function(m2):
    """pdf of reconstructed value"""
    ret = 0
    N = 10000
    m1 = np.linspace(0.2, 1.9, N)
    ret = trans_function(m1[:, None], m2)
    return np.mean(ret, axis=0)


def truth_function(m1):
    """pdf of truth value"""
    ret = 0
    N = 10000
    m2 = np.linspace(0.2, 1.9, N)
    ret = trans_function(m1[:, None], m2)
    return np.mean(ret, axis=1)


def delta_function(delta):
    """pdf of reconstructed value - truth value"""
    N = 10000
    ms_min = 0.2 * 2 + np.abs(delta)  # np.where(delta<0, 0.2+delta, 0.2-delta)
    ms_max = 1.9 * 2 - np.abs(delta)  # np.where(delta<0, 1.9+delta, 1.9-delta)
    # print(delta, ms_min, ms_max)
    ms = (
        np.linspace(0.0 + 1 / 2 / N, 1.0 - 1 / 2 / N, N)[:, None]
        * (ms_max - ms_min)
        + ms_min
    )
    # print(ms, (ms-delta)/2, (ms+delta)/2)
    ret = trans_function((ms - delta) / 2, (ms + delta) / 2)
    return np.mean(ret, axis=0) * (ms_max - ms_min)


def run(name):
    """run detector for data sample"""
    toy = config.data.load_data("data/" + name + ".dat")
    toy_truth, toy_rec = detector(toy)
    toy_rec.savetxt("data/" + name + "_rec.dat", config.get_dat_order())
    toy_truth.savetxt("data/" + name + "_truth.dat", config.get_dat_order())


if __name__ == "__main__":
    run("toy")
    run("phsp")
