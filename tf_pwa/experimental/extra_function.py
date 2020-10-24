import functools

import tensorflow as tf


def extra_function(f0=None, using_numpy=True):
    """Using extra function with numerical differentiation.

    It can be used for numpy function or numba.vectorize function interface.

    >>> import numpy as np
    >>> sin2 = extra_function(np.sin)
    >>> a = tf.Variable([1.0,2.0], dtype="float64")
    >>> with tf.GradientTape(persistent=True) as tape0:
    ...     with tf.GradientTape(persistent=True) as tape:
    ...         b = sin2(a)
    ...     g, = tape.gradient(b, [a,])
    ...
    >>> h, = tape0.gradient(g, [a,])
    >>> assert np.allclose(np.sin([1.0,2.0]), b.numpy())
    >>> assert np.allclose(np.cos([1.0,2.0]), g.numpy())
    >>> assert np.sum(np.abs(-np.sin([1.0,2.0]) - h.numpy())) < 1e-3

    The numerical accuracy is not so well for second derivative.

    """

    def _wrapper(f):
        delta_x = {"float64": 1e-6, "float32": 1e-3}

        @tf.custom_gradient
        def _grad(x, **kwargs):
            if using_numpy and hasattr(x, "numpy"):
                x = x.numpy()
            h = delta_x[x.dtype.name]
            f_u = f(x + h, **kwargs)
            f_d = f(x - h, **kwargs)
            f_0 = f(x, **kwargs)

            def _hess(dg):
                tmp = (f_u + f_d - 2 * f_0) / h / h
                return dg * tmp

            return (f_u - f_d) / 2 / h, _hess

        @tf.custom_gradient
        @functools.wraps(f)
        def _f(x, **kwargs):
            def _g2(dy):
                return dy * _grad(x, **kwargs)

            if using_numpy and hasattr(x, "numpy"):
                x2 = x.numpy()
            else:
                x2 = x
            f_0 = f(x2)
            return f_0, _g2

        return _f

    if f0 is None:
        return _wrapper
    return _wrapper(f0)
