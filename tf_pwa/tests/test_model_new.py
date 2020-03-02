from tf_pwa.model_new import sum_gradient, sum_hessian
from tf_pwa.tensorflow_wrapper import tf

import numpy as np


def test_sum_hessian():
    a = tf.Variable(1.0)
    b = tf.Variable(2.0)

    def f(x):
        return a+b**2+x

    nll, g, h = sum_hessian(f, [1.0, 2.0], [a, b])
    nll1, g1 = sum_gradient(f, [1.0, 2.0], [a, b])
    assert nll == nll1
    assert np.allclose(g, g1)
    assert nll == 13.0
    assert np.allclose(g, np.array([2.0, 8.0]))
    assert np.allclose(h, np.array([[0.0, 0.0], [0.0, 4.0]]))
