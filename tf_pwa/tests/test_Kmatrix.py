import numpy as np

import tf_pwa.amp.Kmatrix
from tf_pwa.amp import get_decay, get_particle, get_relative_p


def check_frac(x, y):
    r = x / y
    assert np.allclose(r, np.mean(r))


def test_a():
    a = get_particle(
        "a",
        mass_list=[2.0],
        width_list=[0.03],
        m1=1.1,
        m2=0.5,
        bw_l=1,
        model="KMatrixSingleChannel",
    )
    b = get_particle("b", J=1, P=-1, mass=2.0, width=0.03, model="BWR")
    c = get_particle("c", J=0, P=-1, mass=1.1, width=0.03, model="one")
    d = get_particle("d", J=0, P=-1, mass=0.5, width=0.03, model="one")
    dec = get_decay(b, [c, d])
    a.init_params()
    b.init_params()

    m = np.array([1.9, 2.0, 2.1])
    q = get_relative_p(m, 1.1, 0.5)
    q0 = get_relative_p(2.0, 1.1, 0.5)
    ay = a.get_amp({"m": m})
    by = b.get_amp({"m": m}, {"|q|": q, "|q0|": q0})
    print(ay / by)
    check_frac(ay.numpy(), by.numpy())


if __name__ == "__main__":
    test_a()
