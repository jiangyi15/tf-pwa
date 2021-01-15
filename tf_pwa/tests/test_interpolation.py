import numpy as np

from tf_pwa.amp import get_particle, interpolation, variable_scope


def test_hist_idx():
    p = get_particle("a", model="hist_idx", min_m=1.0, max_m=3.0, interp_N=20)
    with variable_scope() as vm:
        p.init_params()
        vm.set("a_point_0r", 1.0)
        vm.set("a_point_0i", 0.0)
        vm.set("a_point_1r", 1.0)
        vm.set("a_point_1i", 0.0)
    amp = p(np.array([1.0, 1.001, 2.0, 3.0, 4.0]))
    assert np.allclose(amp[0:2], 1.0)


def test_spline_c_idx():
    p = get_particle(
        "a",
        model="spline_c_idx",
        min_m=1.0,
        max_m=3.0,
        interp_N=20,
        with_bound=True,
    )
    with variable_scope() as vm:
        p.init_params()
        for i in range(20):
            vm.set(f"a_point_{i}r", 1.0)
            vm.set(f"a_point_{i}i", 0.0)
        amp = p(np.array([1.0, 3.0, 2.0]))
    assert np.allclose(amp, 1.0)

    p = get_particle(
        "a",
        model="spline_c_idx",
        polar=False,
        points=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    with variable_scope() as vm:
        p.init_params()
        for i in range(3):
            vm.set(f"a_point_{i}r", i)
            vm.set(f"a_point_{i}i", i)
        amp = p(np.array([1.0, 3.0, 2.0, 4.0]))

    assert np.allclose(amp, [0.0, 1 + 1j, 0.0, 2 + 2j])
