from tf_pwa.cg import get_cg_coef, cg_coef
from tf_pwa.significance import significance
import numpy as np

def test_significance():
    assert np.allclose(significance(0.5, 0.0, 1), 1.0)
    assert np.allclose(significance(0.0, 0.5, 1), 1.0)
    assert np.allclose(significance(1.0, 0.5, 1), 1.0)

def test_get_cg_coef():
    assert np.allclose(get_cg_coef(1, 1, 1, 0, 2, 1), cg_coef(1, 1, 1, 0, 2, 1))
    assert np.allclose(get_cg_coef(2, 1, -1, 0, 2, -1), cg_coef(2, 1, -1, 0, 2, -1))
    assert np.allclose(get_cg_coef(2, 1, -1, 1, 2, 0), cg_coef(2, 1, -1, 1, 2, 0))

