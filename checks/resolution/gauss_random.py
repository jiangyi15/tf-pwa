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


def gauss_random_step(data, decay_chain, r_name, sigma, dat_order):
    angle = cal_helicity_angle(
        data["particle"], decay_chain.standard_topology()
    )
    decay_chain.standard_topology()
    tp_map = decay_chain.topology_map()

    r_particle = tp_map[get_particle(r_name)]

    mass = {}
    for i in data["particle"]:
        mi = data["particle"][i]["m"]
        if i == r_particle:
            mi = mi + tf.random.normal(mi.shape, 0, sigma, dtype=mi.dtype)
        mass[i] = mi

    mask = True
    p4_all = {}
    for i in decay_chain:
        phi = angle[tp_map[i]][tp_map[i.outs[0]]]["ang"]["alpha"]
        theta = angle[tp_map[i]][tp_map[i.outs[0]]]["ang"]["beta"]

        m0 = mass[tp_map[i.core]]
        m1 = mass[tp_map[i.outs[0]]]
        m2 = mass[tp_map[i.outs[1]]]

        mask = mask & (m0 >= m1 + m2)

        p_square = get_relative_p2(m0, m1, m2)

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
            ret[i] = lv.rest_vector(lv.neg(p4_all[tmp]), ret[i])

    ret2 = {}
    mask = tf.expand_dims(mask, -1)
    for i in ret:
        ret2[i] = tf.where(mask, ret[i], data["particle"][tp_map[i]]["p"])
    # print(ret)
    # print({i: data["particle"][tp_map[i]]["p"] for i in decay_chain.outs})

    pi = np.stack([ret2[i] for i in dat_order], axis=1)
    return pi


def gauss_random(data, decay_chain, r_name, sigma, sample_N, dat_order):
    pi_s = []
    for i in range(sample_N):
        pi = gauss_random_step(data, decay_chain, r_name, sigma, dat_order)
        pi_s.append(pi)
    pi = np.concatenate(pi_s, axis=1).reshape((-1, 4))
    return pi


def main():
    sigma = 0.005
    r_name = "R_BC"

    config = ConfigLoader("config.yml")
    sample_N = config.resolution_size

    decays = config.get_decay(False)
    decay_chain = decays.get_decay_chain(r_name)
    data = config.get_data("data_origin")[0]
    phsp = config.get_data("phsp_plot")[0]

    dat_order = config.get_dat_order()

    generate = lambda x: gauss_random(
        x, decay_chain, r_name, sigma, sample_N, dat_order
    )

    pi = generate(data)
    np.savetxt("data/data.dat", pi)
    np.savetxt(
        "data/data_weight.dat", np.ones((pi.shape[0] // len(dat_order),))
    )

    pi = generate(phsp)
    np.save("data/phsp_re.npy", pi)
    np.savetxt(
        "data/phsp_re_weight.dat", np.ones((pi.shape[0] // len(dat_order),))
    )


if __name__ == "__main__":
    main()
