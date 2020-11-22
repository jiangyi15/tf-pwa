import itertools

import numpy as np
import tensorflow as T

from .config import get_config, regist_config


class VarsManager:
    def __init__(self):
        self.trainable_vars = []
        self.variables = {}
        self.variables_value = {}
        self.inner_value = {}

    def get(self, name):
        return self.variables.get(name, None)

    def add(self, name, var, overwrite=True, fix=False, **kwargs):
        if name in self.variables and not overwrite:
            return
        self.variables[name] = var
        if not fix:
            self.trainable_vars.append(name)
        if name not in self.variables_value:
            self.variables_value[name] = var.get_init_value()

    def remove(self, name):
        if name in self.params:
            self.trainable_vars.remove(name)
        if name in self.variables:
            del self.variables[name]
        if name in self.variables_value:
            del self.variables_value[name]

    def get_value(self, name):
        return self.variables_value[name]

    def get_params(self):
        return self.valiables_value

    def forward(self):
        for k, v in self.inner_value.items():
            var = self.variables[k]
            if var.range_ is None:
                self.variables_value[k] = v
                continue
            a, b = var.range_
            if a is not None:
                if b is not None:
                    self.variables_value[k] = T.where(
                        v >= a, T.where(v <= b, v, b), a
                    )
                else:
                    self.variables_value[k] = T.where(v >= a, v, a)
            elif b is not None:
                self.variables_value[k] = T.where(v <= b, v, b)
            else:
                self.variables_value[k] = v

    def set_params(self, value):
        """

        >>> a = VarsManager()
        >>> b = Variable("sss", a)
        >>> a.set_params({"sss": 1.0})

        """
        if isinstance(value, dict):
            for k, v in value.items():
                self.variables_value[k] = v
        elif isinstance(value, (list, tuple, np.ndarray)):
            assert len(value) == len(self.trainable_vars), "{}!={}".format(
                len(value), len(self.trainable_vars)
            )
            for k, v in zip(self.trainable_vars, value):
                self.inner_value[k] = v
            self.forward()
        else:
            raise TypeError("not suported type {}".format(type(value)))


regist_config("vm_new", VarsManager())


class VariableBase:
    def __new__(cls, name, vm=None, **kwargs):
        if vm is None:
            vm = get_config("vm_new")
        var = vm.get(name)
        if var is None:
            var = super().__new__(cls)
        return var

    def __init__(
        self,
        name,
        vm=None,
        shape=(),
        init_value=None,
        range_=None,
        is_complex=False,
        sigma=None,
    ):
        if vm is None:
            vm = get_config("vm_new")
        self.name = name
        self.vm = vm
        self.is_complex = is_complex
        self.init_value = init_value
        self.range_ = range_
        self.sigma = sigma
        self.shape = shape
        vm.add(name, self)

    def get_init_value(self):
        if self.init_value is None:
            return np.random.random()
        return self.init_value

    def __repr__(self):
        return f"<Vairable: {self.name}>"

    def __call__(self):
        return self.vm.get_value(self.name)


def _shape_get(it, fun):
    if isinstance(it, (tuple, list)):
        return [_shape_get(i, fun) for i in it]
    if isinstance(it, dict):
        return {k: _shape_get(v, fun) for k, v in it.items()}
    return fun(it)


def _shape_read(shape):
    shape_range = [list(range(i)) for i in shape]
    shape_idx = itertools.product(*shape_range)
    ret = []
    for i in shape_idx:
        ret.append("_".join([str(j) for j in i]))
    return ret


class ComplexVariable:
    def __new__(cls, name, is_complex=False, **kwargs):
        if not is_complex:
            return VariableBase(name, **kwargs)
        return super().__new__(cls)

    def __init__(self, name, is_complex=False, **kwargs):
        self.name = name
        self.real = VariableBase(name + "r", **kwargs)
        self.imag = VariableBase(name + "i", **kwargs)

    def __call__(self):
        return self.real() + 1.0j * self.imag()


class Variable:
    def __init__(self, name, shape=(), **kwargs):
        if not shape:
            self.name_list = name
            self.variables = ComplexVariable(name, **kwargs)
        else:
            self.name_list = _shape_read(shape)
            self.variables = _shape_get(
                self.name_list, lambda i: ComplexVariable(name + i, **kwargs)
            )

    def __call__(self):
        value = _shape_get(self.variables, lambda i: i())
        return value
