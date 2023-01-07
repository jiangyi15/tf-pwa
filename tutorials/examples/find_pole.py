import numpy as np
import sympy
import tensorflow as tf

from tf_pwa.formula import BWR_dom, create_complex_root_sympy_tfop
from tf_pwa.variable import Variable, VarsManager

m, m0, g0 = sympy.var("m m0 g0")


f = BWR_dom(m, m0, g0, 1, 0.3, 0.3)

g = create_complex_root_sympy_tfop(f, [m0, g0], m, 1.0 + 0.01j)


vm = VarsManager()
a = Variable("a", value=1.0, vm=vm)
b = Variable("b", value=0.1, vm=vm)

with vm.error_trans(np.array([[0.010, 0.0], [0.0, 0.001]])) as pm:
    c = g(a(), b())
    d = tf.math.real(c)
    e = tf.math.imag(c)
print(c, pm.get_error([d, e]))
