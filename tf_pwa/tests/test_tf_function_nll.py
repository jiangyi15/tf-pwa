"""Tests for the compiled (tf.function) NLL gradient used by scipy BFGS."""

import numpy as np
import pytest

from tf_pwa.fit_improve import build_tf_nll_grad
from tf_pwa.tests.test_full import gen_toy, toy_config, toy_config_lazy


@pytest.mark.parametrize("name", ["toy_config", "toy_config_lazy"])
def test_build_tf_nll_grad(request, name):
    configuration = request.getfixturevalue(name)
    fcn = configuration.get_fcn()
    # check=True already compares the compiled callable against the eager path
    f_g = build_tf_nll_grad(fcn)
    y0 = np.asarray(fcn.vm.get_all_val(), dtype=np.float64)
    nll_ref, grad_ref = fcn.nll_grad(y0)
    nll, grad = f_g(y0)
    assert np.allclose(float(nll), float(nll_ref))
    assert np.allclose(np.asarray(grad), np.asarray(grad_ref))


def test_fit_tf_function_nll(toy_config_lazy):
    results = toy_config_lazy.fit(print_init_nll=False, tf_function_nll=True)
    assert np.allclose(results.min_nll, -204.9468493307786)
