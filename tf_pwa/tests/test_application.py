import numpy as np

from tf_pwa.applications import force_pos_def


def test_force_pos():
    a = np.array([[1.0, 0], [0.0, -1.0]])
    force_pos_def(a)
