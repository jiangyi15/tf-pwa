import numpy as np
import tensorflow as tf
import yaml
from detector import trans_function
from scipy.stats import norm

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.data_trans.helicity_angle import (
    HelicityAngle,
    HelicityAngle1,
    generate_p,
)

# loda detector model parameters
with open("detector.yml") as f:
    detector_config = yaml.safe_load(f)


def linear_smear_function(m, m_min, m_max, i, N):
    """generate mass of truth sample and weight"""
    delta_min = m_min - m
    delta_max = m_max - m
    sigma = detector_config["sigma"]
    bias = detector_config["bias"]
    delta_min = np.max(
        [delta_min, -5 * sigma * np.ones_like(delta_min) - bias], axis=0
    )
    delta_max = np.min(
        [delta_max, 5 * sigma * np.ones_like(delta_max) - bias], axis=0
    )
    delta = (
        np.linspace(1 / N / 2, 1 - 1 / N / 2, N)[i] * (delta_max - delta_min)
        + delta_min
    )
    w = trans_function(m + delta, m)
    return m + delta, w


def gauss_interp_function(m, m_min, m_max, i, N):
    """generate mass of truth sample and weight"""
    delta_min = m_min - m
    delta_max = m_max - m
    sigma = detector_config["sigma"]
    bias = detector_config["bias"]
    delta_min = np.max(
        [delta_min, -5 * sigma * np.ones_like(delta_min) - bias], axis=0
    )
    delta_max = np.min(
        [delta_max, 5 * sigma * np.ones_like(delta_max) - bias], axis=0
    )

    from scipy.stats import norm

    prob_min = norm.cdf((delta_min - bias) / sigma)
    prob_max = norm.cdf((delta_max - bias) / sigma)
    prob_interp = (
        np.linspace(1 / (2 * N), 1 - 1 / (2 * N), N)[i] * (prob_max - prob_min)
        + prob_min
    )
    point = norm.ppf(prob_interp)
    delta = point * sigma + bias
    # print(point)
    w = trans_function(m + delta, m) / norm.pdf(point)
    w = np.where(np.isnan(w), 0.0, w)
    return m + delta, w


def legendre_smear_function(m, m_min, m_max, i, N):
    """generate mass of truth sample and weight"""
    delta_min = m_min - m
    delta_max = m_max - m
    sigma = detector_config["sigma"]
    bias = detector_config["bias"]
    delta_min = np.max(
        [delta_min, -5 * sigma * np.ones_like(delta_min) - bias], axis=0
    )
    delta_max = np.min(
        [delta_max, 5 * sigma * np.ones_like(delta_max) - bias], axis=0
    )
    from numpy.polynomial.legendre import leggauss

    point, weight = leggauss(N)
    point, weight = point[i], weight[i]

    delta = (point + 1) / 2 * (delta_max - delta_min) + delta_min

    w = trans_function(m + delta, m) * weight

    w = np.where(np.isnan(w), 0.0, w)
    return m + delta, w


def smear(toy, decay_chain, name, function, idx, N):
    """generate full truth sample and weight"""
    ha = HelicityAngle(decay_chain)

    ms, costheta, phi = ha.find_variable(toy)

    m_min = 0
    m_max = np.inf
    par = None
    for i in decay_chain:
        if str(i.core) == str(name):
            m_min = ms[i.outs[0]] + ms[i.outs[1]]
            par = i.core
        if str(name) in [str(j) for j in i.outs]:
            m_max = ms[i.core]
            for j in i.outs:
                if str(j) == str(name):
                    continue
                m_max = m_max - ms[j]
    m_smear, w = function(ms[par], m_min, m_max, idx, N)
    # print("smear", idx, m_smear, ms[par], m_smear-ms[par], w)
    # exit()

    ms[par] = m_smear

    return ha, ha.build_data(ms, costheta, phi), w


def random_sample(config, decay_chain, toy, smear_method="linear"):
    """generate total truth sample and weight"""
    all_p4 = []
    ws = []

    smear_function = {
        "linear": linear_smear_function,
        "gauss_interp": gauss_interp_function,
        "legendre": legendre_smear_function,
    }[smear_method]

    for i in range(config.resolution_size):
        # decay_chain = [i for i in toy["decay"].keys() if "(p+, p-, pi+, pi-)" in str(i)][0] # config.get_decay(False).get_decay_chain("(B, C)")
        ha, toy_smear, w = smear(
            toy, decay_chain, "BC", smear_function, i, config.resolution_size
        )
        ws.append(w)
        # print(toy_smear)
        # toy_smear = {i: j for i,j in zip(ha.par, toy_smear)}
        # print(toy_smear)
        p4 = [
            toy_smear[i].numpy() for i in config.get_dat_order()
        ]  # (particle, idx, mu)
        all_p4.append(np.stack(p4))  # (resolution, particle, idx, mu)
    ws = np.stack(ws)
    sum_ws = np.sum(ws, axis=0)
    ws = np.where(sum_ws[None, :] != 0, ws, 1e-6)
    sum_ws = np.where(sum_ws == 0, 1.0, sum_ws)
    ws = ws / sum_ws
    return np.stack(all_p4).transpose((2, 0, 1, 3)), ws


def main():
    config = ConfigLoader("config.yml")

    decay_chain = config.get_decay(False).get_decay_chain("BC")

    toy = config.get_data("data_rec")[0]

    ha = HelicityAngle(decay_chain)
    ms, costheta, phi = ha.find_variable(toy)
    dat = ha.build_data(ms, costheta, phi)

    p4, w = random_sample(config, decay_chain, toy, smear_method="legendre")

    np.savetxt("data/data.dat", np.stack(p4).reshape((-1, 4)))
    np.savetxt("data/data_w.dat", np.transpose(w).reshape((-1,)))


if __name__ == "__main__":
    main()
