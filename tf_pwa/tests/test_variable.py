import numpy as np
import tensorflow as tf

from tf_pwa.amp import variable_scope
from tf_pwa.config import set_config
from tf_pwa.function import nll_funciton
from tf_pwa.variable import Bound, Variable, VarsManager, combineVM


def test_variable():
    with variable_scope() as vm:
        m = Variable("R_m", value=2.1)  # trainable is True by default
        g_ls = Variable("A2BR_H", shape=[3], cplx=True)
        fcr = Variable("R_total", cplx=True)
        m1 = Variable("R1_m", value=2.3)
        g_ls1 = Variable("A2BR1_H", shape=[3], cplx=True)
        fcr1 = Variable("R1_total", cplx=True)

        print(g_ls.value)
        print(g_ls())

        m.fixed(2.4)
        g_ls.sameas(g_ls1)
        fcr.r_shareto(fcr1)


def test_variable2():
    set_config("polar", False)
    with variable_scope() as vm:
        b = Variable("R_total", cplx=True)
        assert vm.complex_vars["R_total"] == False
    set_config("polar", True)
    with variable_scope() as vm:
        b = Variable("R_total", cplx=True)
        assert vm.complex_vars["R_total"] == True


def test_combine_vm():
    a = VarsManager("a")
    a_v = Variable("a", vm=a)
    b = VarsManager("b")
    b_v = Variable("b", vm=b)
    com_vm = combineVM(a, b)
    com_vm.set("aa", 3.0)
    assert float(a_v()) == 3.0


def test_minimize():
    with variable_scope() as vm:
        m = Variable("R_m", value=2.1)
        vm.set_bound({"R_m": [-2, 3]})

        def f():
            return m() * m()

        ret = vm.minimize(f)
        print(ret)
        assert np.allclose(m().numpy(), 0.0)


def test_minimize2():
    data = np.linspace(-1.5, 1.5, 10000)
    w = tf.cos(data) + 1
    np.random.seed(2)
    cut = np.random.random(data.shape) * np.max(w) * 1.1 < w
    data = data[cut]

    phsp = np.linspace(-1.5, 1.5, 10000)

    with variable_scope() as vm:
        a = Variable("R_m", cplx=True)
        vm.set_bound({"R_mr": (-2, None)})

        def f(x):
            fx = tf.cos(x) + tf.abs(a())
            return fx

        fcn = nll_funciton(f, data, phsp)
        ret = vm.minimize(fcn)
        print(ret)
        assert abs(float(np.abs(a())) - 1.0) < 0.2
        ret_error = vm.minimize_error(fcn, ret)


def test_minimize():
    with variable_scope() as vm:
        m = Variable("R_m", value=2.1)
        vm.set_bound({"R_m": (None, 3)})

        def f():
            return m() * m()

        from scipy.optimize import minimize as mini

        ret = vm.minimize(
            f, method=lambda g, x: mini(g, x, jac=True, method="L-BFGS-B")
        )
        print(ret)
        assert abs(m().numpy()) < 1e-6
        ret_error = vm.minimize_error(f, ret)


def test_polar():
    with variable_scope() as vm:
        a = Variable("a", cplx=True)
        b = a().numpy()
        vm.rp2xy_all()
        c = a().numpy()
        vm.xy2rp_all()
        d = a().numpy()
        vm.std_polar_all()
        e = vm.get("ai")
        vm.trans_params(True)
        f = a().numpy()
        assert np.allclose(b, c)
        assert np.allclose(c, d)
        assert np.allclose(d, f)
        assert -3.2 < e and e < 3.2


def test_refresh_vars():
    with variable_scope() as vm:
        Variable("a", cplx=True)
        Variable("b", value=1.0)
        Variable("c", value=1.0, range_=[0, 3])

        Variable("b", value=1.0)
        Variable("d", value=1.0, fix=True)
        vm.refresh_vars()
        vm.refresh_vars(bound_dic={"b": (2, 3)})
        vm.refresh_vars(bound_dic={"b": Bound(2, 3)}, init_val={"b": 1.0})


def test_rename():
    with variable_scope() as vm:
        Variable("a", value=1.0)
        vm.rename_var("a", "b")
        assert vm.get("b") == 1.0
        assert "a" not in vm.variables
        a = Variable("d", cplx=True)  # BUG: cannot use "a"
        a.set_value([2, 3])
        vm.rename_var("d", "c", True)
        assert vm.get("cr") == 2
        assert vm.get("ci") == 3
