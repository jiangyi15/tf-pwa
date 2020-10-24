import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
import uproot

from tf_pwa.angle import LorentzVector as lv
from tf_pwa.angle import Vector3 as v3
from tf_pwa.phasespace import PhaseSpaceGenerator

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir + "/..")


def generate_mc(num):
    M_pipm = 0.1395702
    M_pi0 = 0.1349766
    M_Kpm = 0.49368
    M_Dpm = 1.86961
    M_D0 = 1.86483
    M_Dstarpm = 2.01026
    M_Bpm = 5.27926
    M_B0 = 5.27964
    a = PhaseSpaceGenerator(M_Bpm, [M_Dstarpm, M_Dpm, M_Kpm])
    b = PhaseSpaceGenerator(M_Dstarpm, [M_D0, M_pipm])
    p_D1, p_D2, p_K = a.generate(num)
    p_D, p_pi = b.generate(num)
    return [i.numpy() for i in [p_D1, p_D2, p_K, p_D, p_pi]]


def index_bin(x, xi):
    xi1 = np.expand_dims(xi, axis=-1)
    mask = (x[:, 0] < xi1) & (x[:, 1] > xi1)
    idx1, idx2 = np.nonzero(mask)
    idx = np.zeros_like(xi, dtype="int64")
    idx[idx1] = idx2
    return idx


class EffWeight:
    def __init__(self, root_file):
        self.f = uproot.open(root_file)
        self.eff_bin = [
            self.f.get("RegDalitzEfficiency_bin{}".format(i)) for i in range(5)
        ]
        self.x_bins, self.y_bins = self.eff_bin[0].bins  # assert all bins same
        self.values = np.array([i.values for i in self.eff_bin])

    def eff_weight(self, cos_hel, m2_13, m2_23):
        n_histo = (3 - np.floor((cos_hel + 1) / 0.5) + 1).astype("int64")
        x_idx = index_bin(self.x_bins, m2_13)
        y_idx = index_bin(self.y_bins, m2_23)
        return self.values[n_histo, x_idx, y_idx]


def generate_mc_eff(num, eff_file):
    p_D1, p_D2, p_K, p_D, p_pi = generate_mc(num)
    m2_13 = lv.M2(p_D1 + p_K)
    m2_23 = lv.M2(p_D2 + p_K)
    p_D1_r = lv.rest_vector(p_D1 + p_D2, p_D1)
    cos_theta = v3.cos_theta(lv.vect(p_D1_r), lv.vect(p_D))
    eff = EffWeight(eff_file)
    weight = eff.eff_weight(cos_theta, m2_13, m2_23)
    rnd = np.random.random(num)
    mask = weight > rnd
    # plt.hist(cos_theta.numpy()[mask], bins=100)
    # plt.savefig("cos_theta.png")
    p_D1, p_D2, p_K, p_D, p_pi = [
        i[mask] for i in [p_D1, p_D2, p_K, p_D, p_pi]
    ]
    return p_D1, p_D2, p_K, p_D, p_pi


def main(num, eff_file):
    p_D1, p_D2, p_K, p_D, p_pi = generate_mc_eff(num, eff_file)
    p_D, p_pi = [lv.rest_vector(lv.neg(p_D1), i) for i in [p_D, p_pi]]
    data = np.array([p_D1, p_D2, p_K])
    data = np.transpose(data, (1, 0, 2))
    data2 = np.array([p_D, p_pi])
    data2 = np.transpose(data2, (1, 0, 2))
    np.savetxt("data/gen_mc.dat", data.reshape(-1, 4))
    np.savetxt("data/gen_mc_Dstar.dat", data2.reshape(-1, 4))
    bins, x, y, _ = plt.hist2d(lv.M2(p_D1 + p_K), lv.M2(p_D2 + p_K), bins=50)
    plt.clf()
    plt.contourf(
        *np.meshgrid(x[1:] / 2 + x[:-1] / 2, y[1:] / 2 + y[:-1] / 2),
        bins.astype("float"),
    )
    plt.colorbar()
    print(lv.M(p_D1 + p_D2 + p_K))
    plt.savefig("m13_m23.png")


if __name__ == "__main__":
    main(1000000, "data/eff/Efficiency_BmDstpDmKm_run2.root")
