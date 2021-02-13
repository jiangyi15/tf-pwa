#!/usr/bin/env python

"""
Script for generating data for resolution of gauss function.
"""

import numpy as np
import tensorflow as tf

from tf_pwa.amp import get_particle, get_relative_p2
from tf_pwa.angle import LorentzVector as lv
from tf_pwa.angle import Vector3 as v3
from tf_pwa.cal_angle import cal_helicity_angle
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.data import data_shape


def gauss_sample(data, decay_chain, r_name, sigma, dat_order):
    sigma_delta = 5
    sample_N = 30

    def gauss(delta_x):
        return tf.exp(-(delta_x ** 2) / (2 * sigma ** 2))

    angle = cal_helicity_angle(
        data["particle"], decay_chain.standard_topology()
    )
    decay_chain.standard_topology()
    tp_map = decay_chain.topology_map()

    r_particle = tp_map[get_particle(r_name)]

    for i in decay_chain.standard_topology():
        if i.core == r_particle:
            m_min = sum(data["particle"][j]["m"] for j in i.outs)
        print(i.outs, r_particle)
        if any(r_particle == j for j in i.outs):
            m_max = (
                data["particle"][i.core]["m"]
                - sum(data["particle"][j]["m"] for j in i.outs)
                + data["particle"][r_particle]["m"]
            )

    print("min, max: ", m_min, m_max)
    mass = {}
    weights = []
    for i in data["particle"]:
        mi = data["particle"][i]["m"]
        if i == r_particle:
            delta_min = tf.where(
                m_min - mi > -sigma_delta * sigma,
                m_min - mi,
                -sigma_delta * sigma,
            )
            delta_max = tf.where(
                m_max - mi > sigma_delta * sigma,
                sigma_delta * sigma,
                m_max - mi,
            )
            delta_m = (delta_max - delta_min) / (sample_N + 1)
            print("delta_min:", delta_min)
            min_m = mi + delta_min + delta_m / 2
            mi_s = []
            for j in range(sample_N):
                mi_s_i = min_m + delta_m * j
                mi_s.append(mi_s_i)
                weights.append(gauss(mi_s_i - mi))
            mass[i] = tf.stack(mi_s)
        else:
            mass[i] = mi[None, :]

    # print(mass[r_particle], np.mean(mass[r_particle]))

    weights = tf.stack(weights)
    weights = weights / tf.reduce_sum(weights, axis=0)
    data_weights = data.get("weight", tf.ones_like(weights))

    total_weights = weights * data_weights

    print({k: v.shape for k, v in mass.items()})

    mask = True
    p4_all = {}
    for i in decay_chain:
        phi = angle[tp_map[i]][tp_map[i.outs[0]]]["ang"]["alpha"]
        theta = angle[tp_map[i]][tp_map[i.outs[0]]]["ang"]["beta"]

        m0 = mass[tp_map[i.core]]
        m1 = mass[tp_map[i.outs[0]]]
        m2 = mass[tp_map[i.outs[1]]]

        p_square = get_relative_p2(m0, m1, m2)

        print(m0.shape, m1.shape, m2.shape, p_square.shape)

        p = tf.sqrt(tf.where(p_square > 0, p_square, 0))
        pz = p * tf.cos(theta)
        px = p * tf.sin(theta) * tf.cos(phi)
        py = p * tf.sin(theta) * tf.sin(phi)
        E1 = tf.sqrt(m1 * m1 + p * p)
        E2 = tf.sqrt(m2 * m2 + p * p)
        p1 = tf.stack([E1, px, py, pz], axis=-1)
        p2 = tf.stack([E2, -px, -py, -pz], axis=-1)
        p4_all[i.outs[0]] = p1
        p4_all[i.outs[1]] = p2

    print("p shape", {k: v.shape for k, v in p4_all.items()})

    core_boost = {}
    for i in decay_chain:
        if i.core != decay_chain.top:
            core_boost[i.outs[0]] = i.core
            core_boost[i.outs[1]] = i.core
    ret = {}
    for i in decay_chain.outs:
        tmp = i
        ret[i] = p4_all[i]
        while tmp in core_boost:
            tmp = core_boost[tmp]
            # print(i, tmp)
            print(tmp)
            ret[i] = lv.rest_vector(lv.neg(p4_all[tmp]), ret[i])

    ret2 = {}
    mask = tf.expand_dims(mask, -1)
    for i in ret:
        ret2[i] = tf.where(mask, ret[i], data["particle"][tp_map[i]]["p"])

    print("ret2:", {k: v.shape for k, v in ret2.items()})
    # print({i: data["particle"][tp_map[i]]["p"] for i in decay_chain.outs})

    pi = np.stack([ret2[i] for i in dat_order], axis=-2)
    pi = np.transpose(pi, (1, 0, 2, 3))
    total_weights = np.transpose(total_weights.numpy(), (1, 0))
    print(pi.shape)
    return pi, total_weights


def main():
    sigma = 0.005
    sigma_delta = 5
    r_name = "R_BC"
    sample_N = 50

    config = ConfigLoader("config.yml")

    decays = config.get_decay(False)
    decay_chain = decays.get_decay_chain(r_name)
    data = config.get_data("data_origin")[0]
    pi, total_weights = gauss_sample(
        data, decay_chain, "R_BC", sigma, config.get_dat_order()
    )
    np.savetxt("data/data.dat", pi.reshape((-1, 4)))
    np.savetxt("data/data_weight.dat", np.reshape(total_weights, (-1,)))

    data = config.get_data("phsp_plot")[0]
    pi, total_weights = gauss_sample(
        data, decay_chain, "R_BC", sigma, config.get_dat_order()
    )
    np.savetxt("data/phsp_re.dat", pi.reshape((-1, 4)))
    np.savetxt("data/phsp_re_weight.dat", np.reshape(total_weights, (-1,)))


if __name__ == "__main__":
    main()
