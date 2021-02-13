import numpy as np
import tensorflow as tf

from tf_pwa.breit_wigner import BW
from tf_pwa.cal_angle import cal_angle_from_momentum
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.data import data_index, data_mask
from tf_pwa.phasespace import PhaseSpaceGenerator


def gauss(x, sigma):
    a = tf.exp(-(x ** 2) / (2 * sigma ** 2))
    return a


def abs2(x):
    r = tf.math.real(x)
    i = tf.math.imag(x)
    return r * r + i * i


def resolution_bw(m, m0, g0, sigma, m_min, m_max):
    N = 100
    delta_min = -5 * sigma
    delta_max = 5 * sigma
    delta_sigma = (delta_max - delta_min) / (N - 1)
    ret = tf.zeros_like(m)
    zeros = tf.zeros_like(m)
    weights = []
    amps = []
    for i in range(N):
        delta = delta_min + delta_sigma * i
        m_i = delta + m
        amp = abs2(BW(m_i, m0, g0) + 10)
        w = tf.cast(gauss(delta, sigma), m.dtype)
        # print(w)
        w = tf.where((m_i > m_min) & (m_i < m_max), w, zeros)
        # print(m_min, m_max)
        amps.append(amp * w)
        weights.append(w)
    # print(tf.reduce_sum(amps, axis=0), tf.reduce_sum(weights, axis=0))
    return tf.reduce_sum(amps, axis=0) / tf.reduce_sum(weights, axis=0)


def simple_selection(data, weight):
    rnd = tf.random.uniform(weight.shape, dtype=weight.dtype)
    cut = rnd * tf.reduce_max(weight) * 1.2 < weight
    cut_data = data_mask(data, cut)
    return cut_data


def main():
    config = ConfigLoader("config.yml")
    decay = config.get_decay()
    m0 = decay.top.get_mass()
    m1, m2, m3 = [i.get_mass() for i in decay.outs]

    print(m0, m1, m2, m3)

    phsp = PhaseSpaceGenerator(m0, [m1, m2, m3])
    p1, p2, p3 = phsp.generate(100000)

    angle = cal_angle_from_momentum({"B": p1, "C": p2, "D": p3}, decay)
    amp = config.get_amplitude()

    m_idx = config.get_data_index("mass", "R_BC")
    m_BC = data_index(angle, m_idx)
    R_BC = decay.get_particle("R_BC1")
    m_R, g_R = R_BC.get_mass(), R_BC.get_width()

    # import matplotlib.pyplot as plt
    # x = np.linspace(1.3, 1.7, 1000)
    # amp = R_BC.get_amp({"m": x}).numpy()
    # plt.plot(x, np.abs(amp)**2)
    # plt.show()

    print(m_BC, np.mean(m_BC), m_R, g_R)
    amp_s2 = resolution_bw(m_BC, m_R, g_R, 0.005, m1 + m2, m0 - m3)

    print(amp_s2)
    cut_data = simple_selection(angle, amp_s2)

    ps = [data_index(cut_data, ("particle", i, "p")).numpy() for i in "BCD"]
    np.savetxt(
        "data/data_origin.dat", np.transpose(ps, (1, 0, 2)).reshape((-1, 4))
    )

    p1, p2, p3 = phsp.generate(100000)
    np.savetxt(
        "data/phsp.dat", np.transpose([p1, p2, p3], (1, 0, 2)).reshape((-1, 4))
    )

    p1, p2, p3 = phsp.generate(50000)
    np.savetxt(
        "data/phsp_plot.dat",
        np.transpose([p1, p2, p3], (1, 0, 2)).reshape((-1, 4)),
    )


if __name__ == "__main__":
    main()
