from tf_pwa.dfun import *


def test_get_D_matrix_lambda():
    get_D_matrix_lambda(None, 1, (-1, 1), (-1, 1, 1))

    test_angle = {
        "alpha": np.array([1.0, 2.0]),
        "beta": np.array([1.0, 2.0]),
        "gamma": np.array([1.0, 2.0]),
    }

    get_D_matrix_lambda(None, 1, (-1, 1), (-1, 1))
    get_D_matrix_lambda(test_angle, 2, (-2, 2), (-2, 2), (0,))
