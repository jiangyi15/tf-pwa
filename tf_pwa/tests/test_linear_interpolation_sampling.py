import numpy as np

from tf_pwa.generator.breit_wigner import BWGenerator
from tf_pwa.generator.linear_interpolation import (
    LinearInterp,
    LinearInterpImportance,
    interp_sample,
    sample_test_function,
)
from tf_pwa.generator.plane_2d import Interp2D


def test_sampling():
    import matplotlib.pyplot as plt

    fa = sample_test_function
    y, f, scale = interp_sample(fa, 1.0, 2.0, 21, 5000)
    d = np.linspace(1.0, 2.0, 101)
    y, x, _ = plt.hist(
        y,
        bins=100,
        weights=np.ones(y.shape[0])
        * np.sum(fa(d))
        / y.shape[0]
        / d.shape[0]
        * 100,
        label="toy",
    )
    x2 = (x[1:] + x[:-1]) / 2
    # plt.clf()
    # plt.plot(x2, (fa(x2) - y)/np.sqrt(fa(x2)))
    plt.plot(d, f(d) * scale, label="scale intepolation")
    plt.plot(d, f(d), label="intepolation")
    plt.plot(d, fa(d), label="function")
    plt.legend()
    plt.savefig("linear_interpolation_sampling.pdf")
    plt.clf()


def test_importance():
    x = np.linspace(1.0, 2.0, 21)
    f = BWGenerator(1.5, 0.05, 1.0, 2.0)
    g = LinearInterpImportance(f, x)
    y = g.generate(1000)
    y2 = f.generate(1000)
    from scipy.stats import ks_2samp

    a, b = ks_2samp(y, y2)
    assert b > a


def test_integral():
    x = np.linspace(1.0, 2.0, 21)
    f = BWGenerator(1.5, 0.05, 1.0, 2.0)
    g = LinearInterp(x, f(x))
    y = g.integral(x)
    assert 0 == y[0]
    assert np.allclose(y[1:], g.int_step)


def test_plane_2d():
    x = np.linspace(0, 1, 101)
    y = np.linspace(0, 6, 101)
    z = x[:, None] * np.sin(y) + 1
    a = Interp2D(x, y, z)
    data = a.generate(20)
    x, y = data[:, 0], data[:, 1]
    a(x, y)
