import numpy as np
import tensorflow as tf
import yaml
from detector import log_trans_function, trans_function
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

smear_function_table = {}


def register_smear_function(name):
    def _g(f):
        global smear_function_table
        smear_function_table[name] = f
        return f

    return _g


@register_smear_function("linear")
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


@register_smear_function("gauss_interp")
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


@register_smear_function("legendre")
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


@register_smear_function("hermite")
def hermite_smear_function(m, m_min, m_max, i, N):
    """generate mass of truth sample and weight"""
    delta_min = m_min - m
    delta_max = m_max - m
    sigma = detector_config["sigma"]
    bias = detector_config["bias"]
    from numpy.polynomial.hermite import hermgauss

    point, weight = hermgauss(N)
    point, weight = point[i], weight[i]

    # int f(x) exp(-x^2) dx =[t=x/sqrt(2)]= sqrt(2) int f(sqrt(2)t) exp(-t^2/2)dt =
    delta = point * sigma * np.sqrt(2) + bias

    cut = (delta < delta_max) & (delta > delta_min)
    delta = np.where(cut, delta, 0.0)

    w, cut_eff = log_trans_function(m + delta, m)
    w = cut_eff * np.exp(w + point**2) * weight

    w = np.where(cut, w, 0.0)
    w = np.where(np.isnan(w), 0.0, w)
    return m + delta, w


@register_smear_function("hermite2")
def hermite_smear_function(m, m_min, m_max, i, N):
    """generate mass of truth sample and weight"""
    delta_min = m_min - m
    delta_max = m_max - m
    sigma = detector_config["sigma"]
    bias = detector_config["bias"]
    from hermite_truncation import gauss_point

    scale_delta_min = (delta_min - bias) / sigma / np.sqrt(2)
    scale_delta_max = (delta_max - bias) / sigma / np.sqrt(2)

    point, weight = gauss_point(N, scale_delta_min, scale_delta_max)
    point, weight = point[:, i], weight[:, i]

    # int f(x) exp(-x^2) dx =[t=x/sqrt(2)]= sqrt(2) int f(sqrt(2)t) exp(-t^2/2)dt =
    delta = point * sigma * np.sqrt(2) + bias

    cut = (delta < delta_max) & (delta > delta_min)
    delta = np.where(cut, delta, 0.0)

    w, cut_eff = log_trans_function(m + delta, m)
    w = cut_eff * np.exp(w + point**2) * weight

    w = np.where(cut, w, 0.0)
    w = np.where(np.isnan(w), 0.0, w)
    return m + delta, w


@register_smear_function("random")
def random_smear_function(m, m_min, m_max, i, N):
    """generate mass of truth sample and weight"""
    delta_min = m_min - m
    delta_max = m_max - m
    sigma = detector_config["sigma"]
    bias = detector_config["bias"]

    delta = np.random.normal(size=m.shape[0]) * sigma + bias
    cut = (delta >= delta_max) | (delta <= delta_min)
    max_iter = 10
    while np.any(cut) and max_iter > 0:
        point2 = np.random.normal(size=m.shape[0]) * sigma + bias
        point = np.where(cut, point2, delta)
        cut = (delta >= delta_max) | (delta <= delta_min)
        max_iter -= 1

    cut = (delta < delta_max) & (delta > delta_min)
    delta = np.where(cut, delta, 0.0)

    w, cut_eff = log_trans_function(m + delta, m)
    w = cut_eff * np.exp(w + (delta - bias) ** 2 / sigma**2 / 2)

    w = np.where(cut, w, 0.0)
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


def random_sample(
    config, decay_chain, toy, particle="BC", smear_method="linear"
):
    """generate total truth sample and weight"""
    all_p4 = []
    ws = []

    smear_function = smear_function_table[smear_method]

    for i in range(config.resolution_size):
        # decay_chain = [i for i in toy["decay"].keys() if "(p+, p-, pi+, pi-)" in str(i)][0] # config.get_decay(False).get_decay_chain("(B, C)")
        ha, toy_smear, w = smear(
            toy,
            decay_chain,
            particle,
            smear_function,
            i,
            config.resolution_size,
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

    import argparse

    parser = argparse.ArgumentParser(description="sampling for resolution")
    parser.add_argument(
        "--method",
        default="legendre",
        dest="method",
        choices=smear_function_table.keys(),
    )
    parser.add_argument(
        "--particle", default=detector_config["particle"], dest="particle"
    )
    results = parser.parse_args()

    config = ConfigLoader("config.yml")
    config.data.dic["negtive_idx"] = []  # remove negtive idx

    decay_chain = config.get_decay(False).get_decay_chain(results.particle)

    for name in ["data_rec", "bg_rec"]:
        data = config.get_data("deta_rec")
        if data is None:
            continue
        for i, toy in enumerate(data):
            if toy is None:
                continue
            # ha = HelicityAngle(decay_chain)
            # ms, costheta, phi = ha.find_variable(toy)
            # dat = ha.build_data(ms, costheta, phi)

            p4, w = random_sample(
                config, decay_chain, toy, smear_method=results.method
            )
            w = toy.get_weight() * w
            save_name = config.data.dic["data"]
            if isinstance(save_name, list):
                save_name = save_name[i]
            np.savetxt(save_name, np.stack(p4).reshape((-1, 4)))
            save_name = config.data.dic["data_weight"]
            if isinstance(save_name, list):
                save_name = save_name[i]
            np.savetxt(save_name, np.transpose(w).reshape((-1,)))


if __name__ == "__main__":
    main()
