import numpy as np
from .utils import error_print


class NumberError(object):
    """ basic class for propagation of error"""

    def __init__(self, value, error=1.0):
        self._value = value
        self._error = error

    def __repr__(self):
        return error_print(self._value, self._error)

    @property
    def value(self):
        return self._value

    @property
    def error(self):
        return self._error

    def __add__(self, other):
        if isinstance(other, NumberError):
            val = self._value + other._value
            err = np.sqrt(self._error ** 2 + other._error ** 2)
        else:
            val = self._value + other
            err = self._error
        return NumberError(val, err)

    def __sub__(self, other):
        if isinstance(other, NumberError):
            val = self._value - other._value
            err = np.sqrt(self._error ** 2 + other._error ** 2)
        else:
            val = self._value - other
            err = self._error
        return NumberError(val, err)

    def __neg__(self):
        val = -self._value
        err = self._error
        return NumberError(val, err)

    def __mul__(self, other):
        if isinstance(other, NumberError):
            val = self._value * other._value
            err = np.sqrt(
                (self._error * other._value) ** 2
                + (self._value * other._error) ** 2
            )
        else:
            val = self._value * other
            err = self._error * other
        return NumberError(val, err)

    def __truediv__(self, other):
        if isinstance(other, NumberError):
            val = self._value / other._value
            err = (
                np.sqrt(
                    (self._error) ** 2
                    + (self._value * other._error / other._value) ** 2
                )
                / other._value
            )
        else:
            val = self._value / other
            err = self._error / other
        return NumberError(val, err)

    def __pow__(self, other):
        if isinstance(other, NumberError):
            val = self._value ** other._value
            err1 = (
                other._value * self._value ** (other._value - 1) * self._error
            )
            err2 = np.log(other._value) * val * other._error
            err = np.sqrt(err1 ** 2 + err2 ** 2)
        else:
            val = self._value ** other
            err = np.abs(other * self._value ** (other - 1)) * self._error
        return NumberError(val, err)

    def __rpow__(self, other):
        val = other ** self._value
        err = np.log(self._value) * val * self._error
        return NumberError(val, err)

    def log(self):
        val = np.log(self._value)
        err = self._error / np.abs(self._value)
        return NumberError(val, err)

    def exp(self):
        val = np.exp(self._value)
        err = val * self._error
        return NumberError(val, err)

    def apply(self, fun, grad=None, dx=1e-5):
        val = fun(self._value)
        if grad is not None:
            err = np.abs(grad(self._value)) * self._error
        else:
            grad_v = (fun(self._value + dx) - fun(self._value - dx)) / 2 / dx
            err = np.abs(grad_v) * self._error
        return NumberError(val, err)


def cal_err(fun, *args, grad=None, dx=1e-5, kwargs=None):
    kwargs = {} if kwargs is None else kwargs
    value = []
    errs = []
    for i in args:
        if isinstance(i, NumberError):
            value.append(i._value)
            errs.append(i._error)
        else:
            value.append(i)
            errs.append(0.0)
    val = fun(*value, **kwargs)
    if grad is None:
        grad_v = []
        for i, k in enumerate(value):
            value[i] += dx
            val_p = fun(*value, **kwargs)
            value[i] -= 2 * dx
            val_m = fun(*value, **kwargs)
            value[i] = k
            grad_v.append((val_p - val_m) / 2 / dx)
    else:
        grad_v = grad(*value, **kwargs)
    err = np.sqrt(sum((i * j) ** 2 for i, j in zip(grad_v, errs)))
    return NumberError(val, err)
