import matplotlib.pyplot as plt
import numpy as np

from tf_pwa.angle import kine_min_max
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.data import data_index
from tf_pwa.data_trans.dalitz import Dalitz
from tf_pwa.generator.breit_wigner import BWGenerator
from tf_pwa.generator.interp_nd import InterpND


def mass2(pi):
    """mass square"""
    return pi[:, 0] ** 2 - pi[:, 1] ** 2 - pi[:, 2] ** 2 - pi[:, 3] ** 2


def random_rotation(p4):
    """
    random rotation for dalitz plane
    """
    ret = []
    N = p4[0].shape[0]
    alpha = np.random.random(N) * np.pi * 2
    beta = np.arccos(np.random.random(N) * 2 - 1)
    gamma = np.random.random(N) * np.pi * 2
    zeros = np.zeros_like(alpha)
    ones = np.ones_like(alpha)

    # rotation matrix
    rotation_z = np.array(
        [
            [np.cos(alpha), -np.sin(alpha), zeros],
            [np.sin(alpha), np.cos(alpha), zeros],
            [zeros, zeros, ones],
        ]
    )
    rotation_y = np.array(
        [
            [np.cos(beta), zeros, np.sin(beta)],
            [zeros, ones, zeros],
            [-np.sin(beta), zeros, np.cos(beta)],
        ]
    )
    rotation_z2 = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), zeros],
            [np.sin(gamma), np.cos(gamma), zeros],
            [zeros, zeros, ones],
        ]
    )

    rotation_all = np.einsum(
        "ij...,jk...->ik...",
        rotation_z2,
        np.einsum("ij...,jk...->ik...", rotation_y, rotation_z),
    )

    for i in p4:
        E = i[:, 0:1]
        p = i[:, 1:]
        p_r = np.einsum("ij...,...j->...i", rotation_all, p)
        ret.append(np.concatenate([E, p_r], axis=-1))
    return ret


def fill_bound(z):
    ret = z.copy()
    for i in range(z.shape[0]):
        for j in range(z.shape[0]):
            if z[i, j] == 0:
                tmp = 0
                if (i + 1) < z.shape[0] and z[i + 1, j] > 0:
                    tmp += z[i + 1, j]
                    break
                if (i - 1) > 0 and z[i - 1, j] > 0:
                    tmp += z[i - 1, j]
                    break
                if (j + 1) < z.shape[1] and z[i, j + 1] > 0:
                    tmp += z[i, j + 1]
                    break
                if (j - 1) > 0 and z[i, j - 1] > 0:
                    tmp += z[i, j - 1]
                ret[i, j] = tmp
    return ret


# config.set_params("a.json")


def gen_toy1(config):
    toy = config.generate_toy_p(10000)
    plt.clf()
    pi = [toy[i] for i in config.get_dat_order()]
    m12 = mass2(pi[0] + pi[1])
    m23 = mass2(pi[1] + pi[2])
    plt.hist2d(m12, m23, bins=10, cmin=1)
    plt.colorbar()
    plt.savefig("toy1.png")


def gen_toy2(config):

    m0 = config.get_decay().top.mass
    mi = [
        config.get_decay().get_particle(i).mass for i in config.get_dat_order()
    ]
    d = Dalitz(m0, *mi)

    m12_a = np.linspace(mi[0] + mi[1] - 1e-6, m0 - mi[2] + 1e-6, 1000) ** 2
    m23_a = np.linspace(mi[1] + mi[2] - 1e-6, m0 - mi[0] + 1e-6, 1000) ** 2

    # insert more points close to narrow shape
    m12_bw = (
        BWGenerator(1.0, 0.01, 1 - 0.01 * 7, 1 + 0.01 * 7).solve(
            np.linspace(0, 1, 201)
        )
        ** 2
    )  # mi[0]+mi[1]-1e-6, m0-mi[2]+1e-6).solve(np.linspace(0,1, 1000))**2
    m23_bw = (
        BWGenerator(1.0, 0.01, 1 - 0.01 * 7, 1 + 0.01 * 7).solve(
            np.linspace(0, 1, 201)
        )
        ** 2
    )  #  mi[1]+mi[2]-1e-6, m0-mi[0]+1e-6).solve(np.linspace(0,1, 1000))**2
    m12_a = np.concatenate(
        [m12_a[m12_a < m12_bw[0]], m12_bw, m12_a[m12_a > m12_bw[-1]]]
    )
    m23_a = np.concatenate(
        [m23_a[m23_a < m23_bw[0]], m23_bw, m23_a[m23_a > m23_bw[-1]]]
    )

    # eval amplitude in grid data
    m23, m12 = np.meshgrid(m23_a, m12_a)
    m12_f = m12.flatten()
    m23_f = m23.flatten()

    def cal_amp(m12, m23):
        pi = d.generate_p(m12, m23)
        amp = config.eval_amplitude(pi)
        return amp.numpy()

    def batch_cal_amp(m12, m23, max_batch=1000000):
        ret = []
        for i in range((m12.shape[0] + max_batch - 1) // max_batch):
            x = m12[i * max_batch : min((i + 1) * max_batch, m12.shape[0])]
            y = m23[i * max_batch : min((i + 1) * max_batch, m12.shape[0])]
            ret.append(cal_amp(x, y))
        return np.concatenate(ret)

    z = batch_cal_amp(m12_f, m23_f)  # amp.numpy()
    z = np.where(np.isnan(z), np.zeros_like(z), z)
    z = z.reshape(m12_a.shape[0], m23_a.shape[0])  # .transpose((1,0))

    def add_bound(z, m12_a, m23_a, z_fill):
        """fill boundary value with z_fill"""
        z_max = np.max(z)
        z_mean = z_fill
        for i, s in enumerate(m12_a):
            s_min, s_max = kine_min_max(s, m0, mi[0], mi[1], mi[2])
            idx1 = np.digitize(s_min, m23_a[1:-1])
            z[i, idx1] = z_mean
            z[i, idx1 + 1] = z_mean
            idx2 = np.digitize(s_max, m23_a[1:-1])
            z[i, idx2] = z_mean
            z[i, idx2 + 1] = z_mean
        for i, s in enumerate(m23_a):
            s_min, s_max = kine_min_max(s, m0, mi[1], mi[2], mi[0])
            idx1 = np.digitize(s_min, m12_a[1:-1])
            z[idx1, i] = z_mean
            z[idx1 + 1, i] = z_mean
            idx2 = np.digitize(s_max, m12_a[1:-1])
            z[idx2, i] = z_mean
            z[idx2 + 1, i] = z_mean

    # fix boundary effect
    add_bound(z, m12_a, m23_a, np.max(z) / 2)

    f = InterpND([m12_a, m23_a], z)

    # generate toy with importance sampling

    def gen_p(N):
        m = f.generate(N)
        m12 = m[:, 0]
        m23 = m[:, 1]
        pi = d.generate_p(m12, m23)
        p4 = random_rotation(pi)
        return dict(zip(config.get_dat_order(), p4))

    def importance_f(dat):
        pi = [dat[i] for i in config.get_dat_order()]
        m12 = mass2(pi[0] + pi[1])
        m23 = mass2(pi[1] + pi[2])
        # print(m12, m23)
        return f([m12, m23])

    toy = config.generate_toy_p(
        1000000, gen_p=gen_p, importance_f=importance_f
    )

    # plot toy distribution
    plt.clf()
    pi = [toy[i] for i in config.get_dat_order()]
    m12 = mass2(pi[0] + pi[1])
    m23 = mass2(pi[1] + pi[2])
    plt.hist2d(m12, m23, bins=100, cmin=1)
    plt.colorbar()
    plt.savefig("toy2.png")

    # save into file
    np.save("toy.npy", np.stack(pi, axis=-2))


def main():
    config = ConfigLoader("config.yml")
    config.set_params("gen_params.json")  #  = ConfigLoader("config.yml")
    gen_toy2(config)


if __name__ == "__main__":
    main()
