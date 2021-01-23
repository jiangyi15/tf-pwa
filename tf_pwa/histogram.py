import matplotlib.pyplot as plt
import numpy as np


def plot_hist(binning, count, ax=plt, **kwargs):
    n = count.shape[0]
    a = np.zeros((n + 2,))
    b = np.zeros((n + 2,))
    a[:-1] = binning
    a[-1] = binning[-1] + binning[-1] - binning[-2]
    b[1:-1] = count
    return ax.step(a, b, **kwargs)


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

    def draw_bar(self, ax=plt, **kwargs):
        ax.bar(
            (self.binning[:-1] + self.binning[1:]) / 2,
            self.count,
            width=self.binning[1:] - self.binning[:-1],
            **kwargs,
        )

    def draw_pull(self, ax=plt, **kwargs):
        with np.errstate(divide="ignore", invalid="ignore"):
            y_error = np.where(self.error == 0, 0, self.count / self.error)
        ax.bar(
            (self.binning[:-1] + self.binning[1:]) / 2,
            y_error,
            width=self.binning[1:] - self.binning[:-1],
            **kwargs,
        )

    def draw_error(self, ax=plt, fmt="none", **kwargs):
        ax.errorbar(
            (self.binning[:-1] + self.binning[1:]) / 2,
            y=self.count,
            yerr=self.error,
            fmt=fmt,
            color=self._cached_color,
            **kwargs,
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
            count, binning = np.histogram(m, *args, weights=weights, **kwargs)
            count2, _ = np.histogram(m, *args, weights=weights ** 2, **kwargs)
        return Hist1D(binning, count, np.sqrt(count2))

    def get_count(self):
        return np.sum(self.count)
