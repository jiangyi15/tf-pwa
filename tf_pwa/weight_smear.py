import numpy as np

smear_function_table = {}


def register_weight_smear(name):
    def _f(f):
        global smear_function_table
        smear_function_table[name] = f
        return f

    return _f


def get_weight_smear(name):
    if isinstance(name, dict):
        name = name["name"]
    return smear_function_table[name]


@register_weight_smear("Poisson")
def poisson_smear(weight, **kwargs):
    return weight * np.random.poisson(size=weight.shape[0])


@register_weight_smear("Dirichlet")
def dirichlet_smear(weight, **kwargs):
    return weight * np.random.dirichlet(weight) * np.sum(weight)


@register_weight_smear("Gamma")
def gamma_smear(weight, **kwargs):
    return weight * np.random.gamma(1, size=weight.shape[0])
