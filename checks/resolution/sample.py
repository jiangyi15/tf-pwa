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


def smear_function(m, m_min, m_max, i, N):
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


def random_sample(config, decay_chain, toy):
    """generate total truth sample and weight"""
    all_p4 = []
    ws = []

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
    ws = ws / np.sum(ws, axis=0)
    return np.stack(all_p4).transpose((2, 0, 1, 3)), ws


def main():
    config = ConfigLoader("config.yml")

    decay_chain = config.get_decay(False).get_decay_chain("BC")

    toy = config.get_data("data_origin")[0]
    # print(toy)
    # exit()

    ha = HelicityAngle(decay_chain)
    ms, costheta, phi = ha.find_variable(toy)
    dat = ha.build_data(ms, costheta, phi)

    # print(var)
    # print(ha.build_data(*var))
    p4, w = random_sample(config, decay_chain, toy)
    # exit()

    np.savetxt("data/data.dat", np.stack(p4).reshape((-1, 4)))
    np.savetxt(
        "data/data_w.dat", np.transpose(w).reshape((-1,))
    )  # np.ones(p4.reshape((-1,4)).shape[0]))

    toy = config.get_data("phsp_plot")[0]
    p4, w = random_sample(config, decay_chain, toy)

    np.save("data/phsp_plot_re.npy", p4.reshape((-1, 4)))
    np.savetxt(
        "data/phsp_plot_re_w.dat",
        np.transpose(w * toy.get_weight()).reshape((-1,)),
    )  # np.repeat(toy.get_weight(), config.resolution_size))

    # toy = config.get_data("phsp_origin")[0]
    # p4, w = random_sample(config, decay_chain, toy)


#  np.savetxt("data/phsp_re.dat", p4.reshape((-1,4)))
#  np.savetxt("data/phsp_re_w.dat", np.transpose(w * toy.get_weight()).reshape((-1,))) # np.repeat(toy.get_weight(), config.resolution_size))


if __name__ == "__main__":
    main()
