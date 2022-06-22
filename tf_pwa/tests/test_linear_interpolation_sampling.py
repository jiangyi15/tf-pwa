import numpy as np

from tf_pwa.generator.breit_wigner import BWGenerator
from tf_pwa.generator.linear_interpolation import (
    LinearInterpImportance,
    interp_sample,
    sample_test_function,
)


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


def test_importance():
    x = np.linspace(1.0, 2.0, 21)
    f = BWGenerator(1.5, 0.05, 1.0, 2.0)
    g = LinearInterpImportance(f, x)
    y = g.generate(1000)
    y2 = f.generate(1000)
    from scipy.stats import ks_2samp

    a, b = ks_2samp(y, y2)
    assert b > a
