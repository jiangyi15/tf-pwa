import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d


def plot_hist(binning, count, ax=plt, **kwargs):
    n = count.shape[0]
    a = np.zeros((n + 2,))
    b = np.zeros((n + 2,))
    a[:-1] = binning
    a[-1] = binning[-1] + binning[-1] - binning[-2]
    b[1:-1] = count
    return ax.step(a, b, **kwargs)


def interp_hist(binning, y, num=1000, kind="UnivariateSpline"):
    """interpolate data from hostgram into a line"""
    x = (binning[:-1] + binning[1:]) / 2
    if kind == "UnivariateSpline":
        func = UnivariateSpline(x, y, s=2)
    else:
        func = interp1d(x, y, kind=kind, fill_value="extrapolate")
    x_new = np.linspace(
        np.min(binning), np.max(binning), num=num, endpoint=True
    )
    y_new = func(x_new)
    return x_new, y_new


def gauss(x):
    return np.exp(-(x ** 2) / 2) / np.sqrt(2 * np.pi)


def cauchy(x):
    return 1 / (x ** 2 + 1) / np.pi


def epanechnikov(x):
    return np.where((x < 1) & (x > -1), (1 - x ** 2) / 4 * 3, 0)


def uniform(x):
    return np.where((x < 1) & (x > -1), 0.5, 0)


def weighted_kde(m, w, bw, kind="gauss"):
    n = w.shape[0]

    kind_map = {
        "gauss": gauss,
        "cauchy": cauchy,
        "epanechnikov": epanechnikov,
        "uniform": uniform,
    }
    if isinstance(kind, str):
        kernel = kind_map[kind]
    else:
        kernel = kind

    def f(x):
        ret = np.zeros_like(x)
        for i in range(n):
            y = (x - m[i]) / bw[i]
            tmp = w[i] * kernel(y)
            ret += tmp
        return ret

    return f


class Hist1D:
    def __init__(self, binning, count, error=None):
        if error is None:
            error = np.sqrt(count)
        self.binning = binning
        self.count = count
        self.error = error
        self._cached_color = None

    def draw(self, ax=plt, **kwargs):
        a = plot_hist(self.binning, self.count, ax=ax, **kwargs)
        self._cached_color = a[0].get_color()
        return a

    def draw_bar(self, ax=plt, **kwargs):
        return ax.bar(
            self.bin_center,
            self.count,
            width=self.bin_width,
            **kwargs,
        )

    def draw_kde(self, ax=plt, kind="gauss", bin_scale=1.0, **kwargs):
        color = kwargs.pop("color", self._cached_color)
        m = self.bin_center
        bw = self.bin_width * bin_scale
        kde = weighted_kde(m, self.count, bw, kind)
        x = np.linspace(
            self.binning[0], self.binning[-1], self.count.shape[0] * 10
        )
        return ax.plot(x, kde(x), color=color, **kwargs)

    def draw_pull(self, ax=plt, **kwargs):
        with np.errstate(divide="ignore", invalid="ignore"):
            y_error = np.where(self.error == 0, 0, self.count / self.error)
        return ax.bar(
            self.bin_center,
            y_error,
            width=self.bin_width,
            **kwargs,
        )

    def draw_line(self, ax=plt, num=1000, kind="UnivariateSpline", **kwargs):
        x_new, y_new = interp_hist(self.binning, self.count, num, kind)
        return ax.plot(x_new, y_new, **kwargs)

    def draw_error(self, ax=plt, fmt="none", **kwargs):
        color = kwargs.pop("color", self._cached_color)
        return ax.errorbar(
            self.bin_center,
            y=self.count,
            xerr=self.bin_width / 2,
            yerr=self.error,
            fmt=fmt,
            color=color,
            **kwargs,
        )

    @property
    def bin_center(self):
        return (self.binning[:-1] + self.binning[1:]) / 2

    @property
    def bin_width(self):
        return self.binning[1:] - self.binning[:-1]

    def get_bin_weight(self):
        return (self.binning[-1] - self.binning[0]) / (
            self.binning.shape[0] - 1
        )

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Hist1D(self.binning, self.count * other, self.error * other)
        raise NotImplementedError

    __rmul__ = __mul__

    def __add__(self, other):
        assert np.allclose(
            self.binning, other.binning
        ), "need to be the same binning"
        return Hist1D(
            self.binning,
            self.count + other.count,
            np.sqrt(self.error ** 2 + other.error ** 2),
        )

    def __sub__(self, other):
        assert np.allclose(
            self.binning, other.binning
        ), "need to be the same binning"
        return Hist1D(
            self.binning,
            self.count - other.count,
            np.sqrt(self.error ** 2 + other.error ** 2),
        )

    @staticmethod
    def histogram(m, *args, weights=None, **kwargs):
        if weights is None:
            count, binning = np.histogram(m, *args, **kwargs)
            count2, _ = np.histogram(m, *args, **kwargs)
        else:
            weights = np.asarray(weights)
            count, binning = np.histogram(m, *args, weights=weights, **kwargs)
            count2, _ = np.histogram(m, *args, weights=weights ** 2, **kwargs)
        return Hist1D(binning, count, np.sqrt(count2))

    def get_count(self):
        return np.sum(self.count)


class WeightedData(Hist1D):
    def __init__(self, m, *args, weights=None, **kwargs):
        if weights is None:
            weights = np.ones_like(m)
        count, binning = np.histogram(m, *args, weights=weights, **kwargs)
        count2, _ = np.histogram(m, *args, weights=weights ** 2, **kwargs)
        self.value = m
        self.weights = weights
        super().__init__(binning, count, np.sqrt(count2))

    def draw_kde(self, ax=plt, kind="gauss", bin_scale=1.2, **kwargs):
        color = kwargs.pop("color", self._cached_color)
        bw = np.mean(self.bin_width) * bin_scale * np.ones_like(self.value)
        kde = weighted_kde(self.value, self.weights, bw, kind)
        x = np.linspace(
            self.binning[0], self.binning[-1], self.count.shape[0] * 10
        )
        return ax.plot(x, kde(x), color=color, **kwargs)

    def __add__(self, other):
        assert np.allclose(
            self.binning, other.binning
        ), "need to be the same binning"
        ret = WeightedData(
            np.concatenate([self.value, other.value]),
            weights=np.concatenate([self.weights, other.weights]),
        )
        ret.binning = self.binning
        ret.count = self.count + other.count
        ret.error = np.sqrt(self.error ** 2 + other.error ** 2)
        return ret

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            ret = WeightedData(self.value, weights=self.weights * other)
            ret.binning = self.binning
            ret.error = self.error * other
            ret.count = self.count * other
            return ret
        raise NotImplementedError
