import numpy as np


class LinearInterp:
    """
    linear intepolation function for sampling
    """

    def __init__(self, x, y, epsilon=1e-10):
        self.x = x
        self.y = y
        self.N = x.shape[0]
        self.k = np.zeros((self.N - 1,))
        self.b = np.zeros((self.N - 1,))
        self.int_step = np.zeros((self.N - 1,))
        self.epsilon = epsilon
        self.cal_coeffs()

    def cal_coeffs(self):
        self.k = (self.y[1:] - self.y[:-1]) / (self.x[1:] - self.x[:-1])
        self.k = np.where(
            np.abs(self.k) > self.epsilon, self.k, np.zeros_like(self.k)
        )
        self.b = self.y[:-1] - self.k * self.x[:-1]
        int_x = 0.5 * self.k * (
            self.x[1:] ** 2 - self.x[:-1] ** 2
        ) + self.b * (self.x[1:] - self.x[:-1])
        self.int_step = np.cumsum(int_x)

    def generate(self, N):
        x = np.random.random(N) * np.max(self.int_step)
        bin_index = np.digitize(x, self.int_step)
        k = self.k[bin_index]
        b = self.b[bin_index]
        x1 = self.x[1:][bin_index]
        d = x - self.int_step[bin_index]
        y = np.sqrt(b**2 + k * (k * x1**2 + 2 * b * x1 + 2 * d)) - b
        y2 = d + b * x1
        return np.where(k == 0, y2, y) / np.where(k == 0, b, k)

    def __call__(self, x):
        bin_index = np.digitize(x, self.x[1:-1])
        k = self.k[bin_index]
        b = self.b[bin_index]
        return k * x + b


class GenTest:
    def __init__(self, N_max):
        self.N_max = N_max
        self.N_gen = 0
        self.N_total = 0
        self.eff = 0.9

    def generate(self, N):
        self.N_gen = 0
        self.N_total = 0
        while self.N_gen < N:
            test_N = min(int((N - self.N_gen) / self.eff * 1.1), self.N_max)
            self.N_total += test_N
            yield test_N
            self.eff = (self.N_gen + 1) / (self.N_total + 1)  # avoid zero

    def add_gen(self, n_gen):
        # print("add gen")
        self.N_gen = self.N_gen + n_gen

    def set_gen(self, n_gen):
        # print("set gen")
        self.N_gen = n_gen


def interp_sample(f, xmin, xmax, interp_N, N):
    a = np.linspace(
        xmin, xmax, interp_N
    )  # np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.45, 1.47,1.5, 1.53, 1.55, 1.6, 1.7, 1.8, 1.9, 2.0])
    b = f(a)
    f_interp = LinearInterp(a, b)
    all_x = np.array([])
    max_rnd = None
    a = GenTest(100000000)
    for N2 in a.generate(N):
        x, max_rnd_new = interp_sample_once(f, f_interp, N2, max_rnd)
        if max_rnd is None:
            max_rnd = max_rnd_new
        if max_rnd_new > max_rnd:
            cut = np.random.random(all_x.shape[0]) > (
                1 - max_rnd / max_rnd_new
            )
            all_x = all_x[cut]
            a.set_gen(all_x.shape[0])
            max_rnd = max_rnd_new
        a.add_gen(x.shape[0])
        all_x = np.concatenate([all_x, x])
    # print("eff", a.eff, "extra", a.N_gen / N)
    return all_x[:N], f_interp, max_rnd


def interp_sample_once(f, f_interp, N, max_rnd):
    x = f_interp.generate(N)
    w = f(x) / f_interp(x)
    if max_rnd is None:
        max_rnd = np.max(w) * 1.02
    else:
        max_rnd = max(np.max(w) * 1.01, max_rnd)
    rnd = np.random.random(N) * max_rnd
    cut = w > rnd
    return x[cut], max_rnd


def sample_test_function(x):
    return np.abs(1 / (0.005 + (x - 1.5) ** 2)) + 200 * (x - 1.5) ** 2


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    fa = sample_test_function
    y, f, scale = interp_sample(fa, 1.0, 3.0, 21, 50000000)
    d = np.linspace(1.0, 3.0, 10000)
    y, x, _ = plt.hist(
        y,
        bins=10000,
        weights=np.ones(y.shape[0])
        * np.sum(fa(d))
        / y.shape[0]
        / d.shape[0]
        * 10000,
        label="toy",
    )
    x2 = (x[1:] + x[:-1]) / 2
    # plt.clf()
    # plt.plot(x2, (fa(x2) - y)/np.sqrt(fa(x2)))
    plt.plot(d, f(d) * scale, label="scale intepolation")
    plt.plot(d, f(d), label="intepolation")
    plt.plot(d, fa(d), label="function")
    plt.legend()
    plt.savefig("a.pdf")
