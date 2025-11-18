import numpy as np

from tf_pwa.config import create_config
from tf_pwa.generator import BaseGenerator

set_generator, get_generator, register_generator = create_config()


class Simple1DGenerator(BaseGenerator):
    def __init__(self, name, func, params):
        self.name = name
        self.func = func
        self.params = params

    def generate(self, N):
        x = self.func(**self.params, size=(N,))
        return {self.name: x}


class DefaultGenerator(BaseGenerator):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def generate(self, N):
        x = np.ones((N,)) * self.value
        return {self.name: x}


def create_simple_generator(name, params):
    params = params.copy()
    model = params.get("model", "default")
    gen_params = {
        k: v
        for k, v in params.items()
        if k not in ["model", "default", "dtype"]
    }
    if "params" in gen_params:
        gen_params = gen_params["params"]

    model_class = get_generator(model, None)
    if model_class is None:
        if hasattr(np.random, model):
            func = getattr(np.random, model)
            return Simple1DGenerator(name, func, gen_params)
        if model == "default":
            default_var = params["default"]
            return DefaultGenerator(name, value=default_var)
    else:
        return model_class(**gen_params)
    raise ValueError("not support model: {}".format(model))
