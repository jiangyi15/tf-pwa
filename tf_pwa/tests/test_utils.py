import numpy as np

from tf_pwa.utils import *


def test_fit_norml():
    x = np.linspace(-3, 4, 1000)
    w = np.exp(-((x - 0.1) ** 2) / 2 / (1.1 * 1.1))
    x0, xerr = fit_normal(x, w)
    assert np.all(np.abs(x0 - np.array([0.1, 1.1])) < xerr * 3)
    x = np.random.normal(0.1, 1.1, 1000)
    x0, xerr = fit_normal(x)
    assert np.all(np.abs(x0 - np.array([0.1, 1.1])) < xerr * 3)
