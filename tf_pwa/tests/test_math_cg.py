from tf_pwa.cg import cg_coef, get_cg_coef
from tf_pwa.significance import significance


def close_to(x, y, err=1e-5):
    return abs(x - y) < err


def test_significance():
    args = [
        [0.5, 0.0, 1],
        [0.0, 0.5, 1],
        [1.0, 0.5, 1],
        [-2.0, -3.0, 2],
        [-4.0, -4.0, 5],
    ]
    results = [1.0, 1.0, 1.0, 0.9004525966377901, 0.0]
    for i, j in zip(args, results):
        assert close_to(significance(i[0], i[1], i[2]), j)


def test_get_cg_coef():
    assert close_to(get_cg_coef(1, 1, 1, 0, 2, 1), cg_coef(1, 1, 1, 0, 2, 1))
    assert close_to(
        get_cg_coef(2, 1, -1, 0, 2, -1), cg_coef(2, 1, -1, 0, 2, -1)
    )
    assert close_to(get_cg_coef(2, 1, -1, 1, 2, 0), cg_coef(2, 1, -1, 1, 2, 0))
