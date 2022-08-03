import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class InterpND:
    def __init__(self, xs, z):
        self.xs = xs
        self.n_dim = len(xs)
        self.z = z
        self.build_coeffs()
        self.intgral_step()
        self.interp_f = RegularGridInterpolator(xs, z, method="linear")
        # print(self.coeffs)

    def build_coeffs(self):
        self.coeffs = np.zeros((2**self.n_dim, self.n_dim, 2))
        a = [[0, 1]] * self.n_dim
        for i in product(*a):
            idx = 0
            tmp = np.zeros((self.n_dim, 2))
            for j, idx_i in enumerate(i):
                idx = idx + idx_i * 2**j
                if idx_i == 0:
                    tmp[j] = [1, -1]
                else:
                    tmp[j] = [0, 1]
            self.coeffs[idx] = tmp

    def __call__(self, xs):
        bin_index = []
        x = []
        for i, j in zip(xs, self.xs):
            tmp = np.digitize(i, j[1:-1])
            bin_index.append(tmp)
            x.append((i - j[tmp]) / (j[tmp + 1] - j[tmp]))
        # print(xs, bin_index, x)
        z = self.z.flatten()
        ret = 0
        a = [[0, 1]] * self.n_dim
        for i in product(*a):
            tmp = 1
            idx = 0
            for j, (idx_i, delta_idx, xi) in enumerate(zip(bin_index, i, x)):
                # print(idx_i, delta_idx, xi)
                if delta_idx == 0:
                    tmp = tmp * (1 - xi)
                else:
                    tmp = tmp * xi
                idx = (idx + idx_i + delta_idx) * (self.xs[j].shape[0])
            idx = idx // (self.xs[-1].shape[0])
            # print(ret, tmp, idx, z[idx])
            ret = ret + tmp * z[idx]
        # print(z, ret)
        return ret

    def intgral_step(self):
        self.n_bins = 1
        for i in self.xs:
            self.n_bins *= i.shape[0] - 1
        int_shape = [2**self.n_dim] + [(i.shape[0] - 1) for i in self.xs]
        self.int_all = np.zeros(int_shape)

        a = [[slice(0, -1), slice(1, None)]] * self.n_dim
        for i, j in enumerate(product(*a)):
            tmp = self.z.__getitem__(j)
            # print(self.int_all[i], self.z, j, tmp)
            self.int_all[i] = tmp
        self.int_all = self.int_all / (2**self.n_dim)
        self.int_step = np.cumsum(self.int_all.flatten())

    def generate(self, N):
        x = np.sqrt(np.random.random((N, self.n_dim)))
        bin_integal = np.random.random(N) * self.int_step[-1]
        bin_index = np.digitize(bin_integal, self.int_step[:-1])
        p = bin_index // self.n_bins
        bin_index = bin_index % self.n_bins
        coeff = self.coeffs[p]
        xmin = [None] * self.n_dim
        xmax = [None] * self.n_dim
        for i in range(self.n_dim):
            # self.n_dim-1, -1, -1):
            j = self.n_dim - 1 - i
            n = self.xs[j].shape[0] - 1
            idx = bin_index % n
            bin_index = bin_index // n
            xmin[j] = self.xs[j][idx]
            xmax[j] = self.xs[j][idx + 1]
        xmin = np.stack(xmin, axis=-1)
        xmax = np.stack(xmax, axis=-1)
        y = coeff[:, :, 0] + coeff[:, :, 1] * x
        ret = y * (xmax - xmin) + xmin
        # print(xmin, xmax)
        return ret  # [:,::-1]
