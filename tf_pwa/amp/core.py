"""
Basic Amplitude Calculations.
A partial wave analysis process has following structure:

DecayGroup: addition (+)
    DecayChain: multiplication (x)
        Decay, Particle(Propagator)

"""

import contextlib
import functools
import inspect
import warnings
from itertools import combinations
from pprint import pprint

import numpy as np
import sympy as sym

from tf_pwa.breit_wigner import BW, BWR, Bprime, Bprime_q2, to_complex
from tf_pwa.cg import cg_coef
from tf_pwa.config import get_config, regist_config, temp_config
from tf_pwa.data import LazyCall, data_map, data_shape, split_generator
from tf_pwa.dec_parser import load_dec_file
from tf_pwa.dfun import get_D_matrix_lambda
from tf_pwa.einsum import einsum
from tf_pwa.particle import DEFAULT_DECAY, BaseParticle, Decay
from tf_pwa.particle import DecayChain as BaseDecayChain
from tf_pwa.particle import DecayGroup as BaseDecayGroup
from tf_pwa.particle import (
    _spin_int,
    _spin_range,
    cp_charge_group,
    split_particle_type,
)
from tf_pwa.tensorflow_wrapper import tf
from tf_pwa.variable import Variable, VarsManager

# from pysnooper import snoop


PARTICLE_MODEL = "particle_model"
regist_config(PARTICLE_MODEL, {})
DECAY_MODEL = "decay_model"
regist_config(DECAY_MODEL, {})
DECAY_CHAIN_MODEL = "decay_chain_model"
regist_config(DECAY_CHAIN_MODEL, {})


def register_particle(name=None, f=None):
    """register a particle model

    :params name: model name used in configuration
    :params f: Model class
    """

    def regist(g):
        if name is None:
            my_name = g.__name__
        else:
            my_name = name
        config = get_config(PARTICLE_MODEL)
        if my_name in config:
            warnings.warn("Override model {}".format(my_name))
        config[my_name] = g
        g.model_name = my_name
        return g

    if f is None:
        return regist
    return regist(f)


def register_decay(name=None, num_outs=2, f=None):
    """register a decay model

    :params name: model name used in configuration
    :params f: Model class
    """

    def regist(g):
        if name is None:
            my_name = g.__name__
        else:
            my_name = name
        config = get_config(DECAY_MODEL)
        id_ = (num_outs, my_name)
        if id_ in config:
            warnings.warn("Override deccay model {}".format(my_name))
        config[id_] = g
        g.model_name = my_name
        return g

    if f is None:
        return regist
    return regist(f)


def register_decay_chain(name=None, f=None):
    """register a decay model

    :params name: model name used in configuration
    :params f: Model class
    """

    def regist(g):
        if name is None:
            my_name = g.__name__
        else:
            my_name = name
        config = get_config(DECAY_CHAIN_MODEL)
        id_ = my_name
        if id_ in config:
            warnings.warn("Override deccay model {}".format(my_name))
        config[id_] = g
        g.model_name = my_name
        return g

    if f is None:
        return regist
    return regist(f)


regist_particle = register_particle
regist_decay = register_decay


def get_particle_model(name):
    all_model = get_config(PARTICLE_MODEL)
    return all_model.get(name, None)


def get_particle_model_name(p):
    all_model = get_config(PARTICLE_MODEL)
    for k, v in all_model.items():
        if type(p) is v:
            return k
    return str(type(p))


def get_particle(*args, model="default", **kwargs):
    """method for getting particle of model"""
    if isinstance(model, dict):
        model_class = trans_model(model)
    else:
        model_class = get_particle_model(model)
    if model_class is None:
        warnings.warn(
            "No model named {} found, use default instead.".format(model)
        )
        model_class = get_particle_model("default")
    ret = model_class(*args, **kwargs)
    ret.model_name = model
    return ret


def trans_model(model):
    expr = model.get("expr")
    expr = sym.simplify(expr)
    var = {str(k): str(k) for k in expr.free_symbols}
    var.update(model.get("where", {}))
    model_name = []
    for k, v in var.items():
        if isinstance(v, str):
            model_name.append((k, v))
    assert len(model_name) == 1
    expr = sym.simplify(expr)
    var_name, name = model_name.pop()
    expr2 = expr.subs({k: v for k, v in var.items() if k != var_name})
    assert len(expr2.free_symbols) == 1, str(expr2)
    fun = sym.lambdify((var_name,), expr2, "tensorflow")
    base_model = get_particle_model(name)

    class _TempModel(base_model):
        _from_trans = True

        def get_amp(self, *args, **kwargs):
            amp = super().get_amp(*args, **kwargs)
            return fun(amp)

    return _TempModel


def get_decay(core, outs, **kwargs):
    """method for getting decay of model"""
    num_outs = len(outs)

    prod_params = {}
    for i in outs:
        prod_params.update(getattr(i, "production_params", {}))

    decay_params = getattr(core, "decay_params", {})

    new_kwargs = {**prod_params, **decay_params, **kwargs}

    model = new_kwargs.get("model", "default")
    id_ = (num_outs, model)

    return get_config(DECAY_MODEL)[id_](core, outs, **new_kwargs)


def get_decay_chain(decays, **kwargs):
    """method for getting decay of model"""
    decay_params = {}
    for i in decays:
        decay_params.update(getattr(i, "decay_chain_params", {}))

    new_kwargs = {**decay_params, **kwargs}

    model = new_kwargs.pop("model", "default")

    return get_config(DECAY_CHAIN_MODEL)[model](decays, **new_kwargs)


def data_device(data):
    def get_device(dat):
        if hasattr(dat, "device"):
            return dat.device
        return None

    pprint(data_map(data, get_device))
    return data


def get_name(self, names):
    name = (
        (str(self) + "_" + names)
        .replace(":", "/")
        .replace("+", ".")
        .replace(",", "")
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "")
    )
    return name


def _add_var(self, names, is_complex=False, shape=(), **kwargs):
    name = get_name(self, names)
    return Variable(name, shape, is_complex, **kwargs)


class AmpBase(object):
    """Base class for amplitude"""

    def add_var(self, names, is_complex=False, shape=(), **kwargs):
        """
        default add_var method
        """
        if not hasattr(self, "_variables_map"):
            self._variables_map = {}
        name = self.get_variable_name(names)
        var = Variable(name, shape, is_complex, **kwargs)
        self._variables_map[names] = var
        return var

    def get_var(self, name):
        return getattr(self, "_variables_map", {}).get(name)

    def get_variable_name(self, name=""):
        return get_name(self, name)

    def amp_shape(self):
        raise NotImplementedError

    def get_factor_variable(self):
        return []


@contextlib.contextmanager
def variable_scope(vm=None):
    """variabel name scope"""
    if vm is None:
        vm = VarsManager(dtype=get_config("dtype"))
    with temp_config("vm", vm):
        yield vm


def simple_deepcopy(dic):
    if isinstance(dic, dict):
        return {k: simple_deepcopy(v) for k, v in dic.items()}
    if isinstance(dic, list):
        return [simple_deepcopy(v) for v in dic]
    if isinstance(dic, tuple):
        return tuple([simple_deepcopy(v) for v in dic])
    return dic


def simple_cache_fun(f):
    name = "simple_cached_" + f.__name__

    @functools.wraps(f)
    def g(self, *args, **kwargs):
        if not hasattr(self, name):
            setattr(self, name, f(self, *args, **kwargs))
        return getattr(self, name)

    return g


def get_relative_p(m_0, m_1, m_2):
    """relative momentum for 0 -> 1 + 2"""
    M12S = m_1 + m_2
    M12D = m_1 - m_2
    if hasattr(M12S, "dtype"):
        m_0 = tf.convert_to_tensor(m_0, dtype=M12S.dtype)
    m_eff = tf.where(m_0 > M12S, m_0, M12S)
    p = (m_eff - M12S) * (m_eff + M12S) * (m_eff - M12D) * (m_eff + M12D)
    # if p is negative, which results from bad data, the return value is 0.0
    # print("p", tf.where(p==0), m_0, m_1, m_2)
    return tf.sqrt(p) / (2 * m_eff)


def get_relative_p2(m_0, m_1, m_2):
    """relative momentum for 0 -> 1 + 2"""
    M12S = m_1 + m_2
    M12D = m_1 - m_2
    if hasattr(M12S, "dtype"):
        m_0 = tf.convert_to_tensor(m_0, dtype=M12S.dtype)
    # m_eff = tf.where(m_0 > M12S, m_0, M12S)
    p = (m_0 - M12S) * (m_0 + M12S) * (m_0 - M12D) * (m_0 + M12D)
    # if p is negative, which results from bad data, the return value is 0.0
    # print("p", tf.where(p==0), m_0, m_1, m_2)
    return p / (2 * m_0) ** 2


def _ad_hoc(m0, m_max, m_min):
    r"""ad-hoc formula

    .. math::
        m_0^{eff} = m^{min} + \frac{m^{max} - m^{min}}{2}(1+tanh \frac{m_0 - \frac{m^{max} + m^{min}}{2}}{m^{max} - m^{min}})

    """
    k = (m_max - m_min) / 2
    m_eff = k * (1 + tf.tanh((2 * m0 - (m_max + m_min)) / k / 4))
    return m_eff + m_min


@regist_particle("BWR")
@regist_particle("default")
class Particle(BaseParticle, AmpBase):
    """
        .. math::
            R(m) = \\frac{1}{m_0^2 - m^2 - i m_0 \\Gamma(m)}

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> from tf_pwa.utils import plot_particle_model
        >>> axis = plot_particle_model("BWR")

    """

    def __init__(self, *args, running_width=True, bw_l=None, **kwargs):
        super(Particle, self).__init__(*args, **kwargs)
        self.running_width = running_width
        self.bw_l = bw_l

    def init_params(self):
        self.d = 3.0
        if self.mass is None:
            self.mass = self.add_var("mass", fix=True)
            # print("$$$$$",self.mass)
        else:
            if not isinstance(self.mass, Variable):
                self.mass = self.add_var("mass", value=self.mass, fix=True)
        if self.width is not None:
            if not isinstance(self.width, Variable):
                self.width = self.add_var("width", value=self.width, fix=True)

    def is_fixed_shape(self):
        for k, v in self.__dict__.items():
            if isinstance(v, Variable):
                if not v.is_fixed():
                    return False
        return True

    def get_amp(self, data, data_c, **kwargs):
        mass = self.get_mass()
        width = self.get_width()
        if width is None:
            return tf.ones_like(data["m"])
        if not self.running_width:
            ret = BW(data["m"], mass, width)
        else:
            q = data_c["|q|"]
            q0 = data_c["|q0|"]
            if self.bw_l is None:
                decay = self.decay[0]
                self.bw_l = min(decay.get_l_list())
            ret = BWR(data["m"], mass, width, q, q0, self.bw_l, self.d)
            # ret = tf.where(q0 > 0, ret, tf.zeros_like(ret))
            # ret = tf.where(q > 0, ret, tf.zeros_like(ret))
        return ret

    def __call__(self, m):
        mass = self.get_mass()
        m1 = self.decay[0].outs[0].get_mass()
        m2 = self.decay[0].outs[1].get_mass()
        q = get_relative_p(m, m1, m2)
        q0 = get_relative_p(mass, m1, m2)
        return self.get_amp(
            {"m": m}, {"|q|": q, "|q|2": q**2, "|q0|": q0, "|q0|2": q0**2}
        )

    def amp_shape(self):
        return ()

    def get_mass(self):
        if self.mass is None:
            warnings.warn(
                f"The mass of {self} is None, may be you should calculate amplitude first to infer mass"
            )
        if callable(self.mass):
            return self.mass()
        return self.mass

    def get_width(self):
        if callable(self.width):
            return self.width()
        return self.width

    def get_factor(self):
        return None

    def get_sympy_var(self):
        return sym.var("m m0 g0 m1 m2")

    def get_subdecay_mass(self, idx=0):
        return [i.get_mass() for i in self.decay[idx].outs]

    def get_num_var(self):
        mass = self.get_mass()
        width = self.get_width()
        m1, m2 = self.get_subdecay_mass()
        return mass, width, m1, m2

    def solve_pole(self):
        mass = self.get_mass()
        width = self.get_width()
        if width is None:
            raise NotImplemented
        from tf_pwa.formula import create_complex_root_sympy_tfop

        var = self.get_sympy_var()
        f = self.get_sympy_dom(*var)
        g = create_complex_root_sympy_tfop(
            f, var[1:], var[0], float(mass) - sym.I * float(width) / 2
        )
        return g(*self.get_num_var())

    def get_sympy_dom(self, m, m0, g0, m1=None, m2=None):
        if self.get_width() is None:
            raise NotImplemented
        from tf_pwa.formula import BW_dom, BWR_dom

        if not self.running_width or m1 is None or m2 is None:
            return BW_dom(m, m0, g0)
        else:
            if self.bw_l is None:
                decay = self.decay[0]
                self.bw_l = min(decay.get_l_list())
            return BWR_dom(m, m0, g0, self.bw_l, m1, m2)


@regist_particle("x")
class ParticleX(Particle):
    """simple particle model for mass, (used in expr)

        .. math::
            R(m) = m

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> from tf_pwa.utils import plot_particle_model
        >>> axis = plot_particle_model("x")

    """

    def __call__(self, m):
        return self.get_amp({"m": m})

    def get_amp(self, data, *args, **kwargs):
        m = data["m"]
        zeros = tf.zeros_like(m)
        return tf.complex(m, zeros)


class SimpleResonances(Particle):
    def __init__(self, *args, **kwargs):
        self.params = {}
        super(SimpleResonances, self).__init__(*args, **kwargs)

    def __call__(self, m, m0=None, g0=None, q=None, q0=None, **kwargs):
        raise NotImplementedError

    def get_amp(self, *args, **kwargs):
        m = args[0]["m"]
        q, q0 = None, None
        if len(args) >= 2:
            q = args[1].get("|q|", 1.0)
            q0 = args[1].get("|q0|", 1.0)
        m0 = self.get_mass()
        g0 = self.get_width()
        return self(m, m0=m0, g0=g0, q=q, q0=q0, **kwargs)


class FloatParams(float):
    pass


def simple_resonance(name, fun=None, params=None):
    """convert simple fun f(m) into a resonances model

    :params name: model name used in configuration
    :params fun: Model function
    :params params: arguments name list for parameters

    """

    if params is None:
        params = {}

    def _wrapper(f):
        argspec = inspect.getfullargspec(f)
        args = argspec.args
        if argspec.defaults is None:
            defaults = {}
        else:
            defaults = dict(zip(argspec.args[::-1], argspec.defaults[::-1]))

        @register_particle(name)
        class _R(SimpleResonances):
            def init_params(self):
                if "m0" in argspec.args and "g0" in argspec.args:
                    super(_R, self).init_params()
                self.params = {}
                for i in argspec.args:
                    tp = argspec.annotations.get(i, None)
                    if i in params or tp is FloatParams:
                        val = getattr(self, i, defaults.get(i, None))
                        if val is None:
                            self.params[i] = self.add_var(i)
                        else:
                            self.params[i] = self.add_var(
                                i, value=val, fix=True
                            )

            def is_fixed_shape(self):
                ret = super().is_fixed_shape()
                for k, v in self.params.items():
                    ret = ret and v.is_fixed()
                return ret

            def __call__(self, m, **kwargs):
                my_kwargs = {}
                for i in argspec.args:
                    if i in kwargs:
                        my_kwargs[i] = kwargs[i]
                    elif i in self.params:
                        my_kwargs[i] = self.params[i]()
                    elif hasattr(self, i):
                        my_kwargs[i] = getattr(self, i)
                ret = f(m, **my_kwargs)
                return tf.cast(ret, tf.complex128)

            __call__.__doc__ = f.__doc__

        _R.get_amp.__doc__ = f.__doc__
        return _R

    if fun is None:
        return _wrapper
    return _wrapper(fun)


class AmpDecay(Decay, AmpBase):
    """base class for decay with amplitude"""

    def amp_shape(self):
        ret = [len(self.core.spins)]
        for i in self.outs:
            ret.append(len(i.spins))
        return tuple(ret)

    # @simple_cache_fun
    def amp_index(self, base_map):
        ret = [base_map[self.core]]
        for i in self.outs:
            ret.append(base_map[i])
        return ret


@regist_decay("default")
@regist_decay("gls-bf")
class HelicityDecay(AmpDecay):
    r"""default decay model


    The total amplitude is

    .. math::
        A = H_{\lambda_{B},\lambda_{C}}^{A \rightarrow B+C} D^{J_A*}_{\lambda_{A}, \lambda_{B}-\lambda_{C}} (\varphi,\theta,0)

    The helicity coupling is

    .. math::
        H_{\lambda_{B},\lambda_{C}}^{A \rightarrow B+C} =
        \sum_{ls} g_{ls} \sqrt{\frac{2l+1}{2 J_{A}+1}} \langle l 0; s \delta|J_{A} \delta\rangle \langle J_{B} \lambda_{B} ;J_{C} -\lambda_{C} | s \delta \rangle q^{l} B_{l}'(q, q_0, d)

    The fit parameters is :math:`g_{ls}`

    There are some options

    (1). `has_bprime=False` will remove the :math:`B_{l}'(q, q_0, d)` part.

    (2). `has_barrier_factor=False` will remove the :math:`q^{l} B_{l}'(q, q_0, d)` part.

    (3). `barrier_factor_norm=True` will replace :math:`q^l` with :math:`(q/q_{0})^l`

    (4). `below_threshold=True` will replace the mass used to calculate :math:`q_0` with

        .. math::
            m_0^{eff} = m^{min} + \frac{m^{max} - m^{min}}{2}(1+tanh \frac{m_0 - \frac{m^{max} + m^{min}}{2}}{m^{max} - m^{min}})

    (5). `l_list=[l1, l2]` and `ls_list=[[l1, s1], [l2, s2]]` options give the list of all possible LS used in the decay.

    """

    def __init__(
        self,
        *args,
        has_barrier_factor=True,
        l_list=None,
        barrier_factor_mass=False,
        has_bprime=True,
        aligned=False,
        allow_cc=True,
        ls_list=None,
        barrier_factor_norm=False,
        params_polar=None,
        below_threshold=False,
        **kwargs
    ):
        super(HelicityDecay, self).__init__(*args, **kwargs)
        self.has_barrier_factor = has_barrier_factor
        self.l_list = l_list
        self.barrier_factor_mass = barrier_factor_mass
        self.has_bprime = has_bprime
        self.aligned = aligned
        self.allow_cc = allow_cc
        self.single_gls = False
        self.ls_index = None
        self.total_ls = None
        self.barrier_factor_norm = barrier_factor_norm
        self.below_threshold = below_threshold
        self.ls_list = None
        if ls_list is not None:
            self.ls_list = tuple([tuple(i) for i in ls_list])
        self.params_polar = params_polar
        self.mask_factor = False

    def check_valid_jp(self):
        if len(self.get_ls_list()) == 0:
            if not self.p_break:
                raise ValueError(
                    """invalid spin parity for {}, maybe you should set `p_break: True` for weak decay""".format(
                        self
                    )
                )
            raise ValueError("invalid spin parity for {}".format(self))

    def set_ls(self, ls):
        if self.total_ls is None:
            self.total_ls = self.get_ls_list()
        self.ls_list = tuple([tuple(i) for i in ls])
        self.single_gls = len(ls) == 1
        # print(self, "total_ls: ", self.total_ls)
        total_ls = self.total_ls
        if len(total_ls) == len(ls):
            self.ls_index = None
            return
        self.ls_index = []
        for i in self.ls_list:
            self.ls_index.append(total_ls.index(i))

    def init_params(self):
        self.d = 3.0
        ls = self.get_ls_list()
        self.g_ls = self.add_var(
            "g_ls", is_complex=True, polar=self.params_polar, shape=(len(ls),)
        )
        try:
            self.g_ls.set_fix_idx(fix_idx=0, fix_vals=(1.0, 0.0))
        except Exception as e:
            print(e, self, self.get_ls_list())

    def get_factor_variable(self):
        return [(self.g_ls,)]

    def get_factor(self):
        return self.get_g_ls()

    def _get_particle_mass(self, p, data, from_data=False):
        if from_data and p in data:
            return data[p]["m"]
        if p.mass is None:
            p.mass = tf.reduce_mean(data[p]["m"])
        return p.get_mass()

    def get_relative_momentum(self, data, from_data=False):
        """"""

        _get_mass = lambda p: self._get_particle_mass(p, data, from_data)

        m0 = _get_mass(self.core)
        m1 = _get_mass(self.outs[0])
        m2 = _get_mass(self.outs[1])
        if self.below_threshold:
            m3 = _get_mass(
                [i for i in self.core.creators[0].outs if i != self.core][0]
            )
            m_eff = _ad_hoc(
                m0, _get_mass(self.core.creators[0].core) - m3, m1 + m2
            )
            m0 = tf.where(m0 < m1 + m2, m_eff, m0)
        return get_relative_p(m0, m1, m2)

    def get_relative_momentum2(self, data, from_data=False):
        """"""

        _get_mass = lambda p: self._get_particle_mass(p, data, from_data)

        m0 = _get_mass(self.core)
        m1 = _get_mass(self.outs[0])
        m2 = _get_mass(self.outs[1])
        if self.below_threshold:
            m3 = _get_mass(
                [i for i in self.core.creators[0].outs if i != self.core][0]
            )
            m_eff = _ad_hoc(
                m0, _get_mass(self.core.creators[0].core) - m3, m1 + m2
            )
            m0 = tf.where(m0 < m1 + m2, m_eff, m0)
        ret = get_relative_p2(m0, m1, m2)
        return ret

    def get_cg_matrix(self, out_sym=False):
        ls = self.get_ls_list()
        return self._get_cg_matrix(ls, out_sym=out_sym)

    @functools.lru_cache()
    def _get_cg_matrix(self, ls, out_sym=False):  # CG factor inside H
        """
        [(l,s),(lambda_b,lambda_c)]

        .. math::
          \\sqrt{\\frac{ 2 l + 1 }{ 2 j_a + 1 }}
          \\langle j_b, j_c, \\lambda_b, - \\lambda_c | s, \\lambda_b - \\lambda_c \\rangle
          \\langle l, s, 0, \\lambda_b - \\lambda_c | j_a, \\lambda_b - \\lambda_c \\rangle
        """
        m = len(ls)
        ja = self.core.J
        jb = self.outs[0].J
        jc = self.outs[1].J
        n = len(self.outs[0].spins), len(
            self.outs[1].spins
        )  # _spin_int(2 * jb + 1), _spin_int(2 * jc + 1)
        ret = np.zeros(shape=(m, *n))
        sqrt = np.sqrt
        my_cg_coef = cg_coef
        if out_sym:
            from sympy.physics.quantum.cg import CG

            sint = lambda x: sym.simplify(_spin_int(x * 2)) / 2
            sqrt = lambda x: sym.sqrt(sint(x))

            def my_cg_coef(a, b, c, d, e, f):
                return CG(*list(map(sint, [a, c, b, d, e, f]))).doit()

            ret = ret.tolist()
        for i, ls_i in enumerate(ls):
            l, s = ls_i
            for i1, lambda_b in enumerate(
                self.outs[0].spins
            ):  # _spin_range(-jb, jb)):
                for i2, lambda_c in enumerate(
                    self.outs[1].spins
                ):  # _spin_range(-jc, jc)):
                    ret[i][i1][i2] = (
                        sqrt(2 * l + 1)
                        / sqrt(2 * ja + 1)
                        * my_cg_coef(
                            jb, jc, lambda_b, -lambda_c, s, lambda_b - lambda_c
                        )
                        * my_cg_coef(
                            l,
                            s,
                            0,
                            lambda_b - lambda_c,
                            ja,
                            lambda_b - lambda_c,
                        )
                    )
        return ret

    def build_ls2hel_eq(self):
        cg_matrix = self.get_cg_matrix(out_sym=True)
        gls = []
        for l, s in self.get_ls_list():
            gls.append(sym.Symbol("g_{}_{}".format(l, s)))
        hel = []
        eqs = []
        for ib, lb in enumerate(self.outs[0].spins):
            for ic, lc in enumerate(self.outs[1].spins):
                tmp = sym.Symbol("H_{}_{}".format(lb, lc))
                rhs = 0
                for idx, gi in enumerate(gls):
                    rhs = rhs + cg_matrix[idx][ib][ic] * gi
                if rhs == 0:
                    continue
                eq = sym.Eq(tmp, sym.simplify(rhs))
                hel.append(tmp)
                eqs.append(eq)
        return [gls, hel, eqs]

    def build_simple_data(self):
        data_p = {self.core: {"m": self.core.get_mass()}}
        data = {}
        zero = np.array(0.0)
        for i in self.outs:
            data_p[i] = {"m": i.get_mass()}
            data[i] = {"ang": {"alpha": zero, "beta": zero, "gamma": zero}}
        return {"data": data, "data_p": data_p}

    def get_helicity_amp(self, data, data_p, **kwargs):
        m_dep = self.get_ls_amp(data, data_p, **kwargs)
        cg_trans = tf.cast(self.get_cg_matrix(), m_dep.dtype)
        n_ls = len(self.get_ls_list())
        m_dep = tf.reshape(m_dep, (-1, n_ls, 1, 1))
        cg_trans = tf.reshape(
            cg_trans, (n_ls, len(self.outs[0].spins), len(self.outs[1].spins))
        )
        H = tf.reduce_sum(m_dep * cg_trans, axis=1)
        # print(n_ls, cg_trans, self, m_dep.shape) # )data_p)
        if self.allow_cc:
            all_data = kwargs.get("all_data", {})
            charge = all_data.get("charge_conjugation", None)
            if charge is not None:
                H = tf.where(
                    charge[..., None, None] > 0, H, H[..., ::-1, ::-1]
                )
        ret = tf.reshape(
            H, (-1, 1, len(self.outs[0].spins), len(self.outs[1].spins))
        )
        return ret

    def get_angle_helicity_amp(self, data, data_p, **kwargs):
        m_dep = self.get_angle_ls_amp(data, data_p, **kwargs)
        cg_trans = tf.cast(self.get_cg_matrix(), m_dep.dtype)
        n_ls = len(self.get_ls_list())
        m_dep = tf.reshape(m_dep, (-1, n_ls, 1, 1))
        cg_trans = tf.reshape(
            cg_trans, (n_ls, len(self.outs[0].spins), len(self.outs[1].spins))
        )
        H = tf.reduce_sum(m_dep * cg_trans, axis=1)
        # print(n_ls, cg_trans, self, m_dep.shape) # )data_p)
        if self.allow_cc:
            all_data = kwargs.get("all_data", {})
            charge = all_data.get("charge_conjugation", None)
            if charge is not None:
                H = tf.where(
                    charge[..., None, None] > 0, H, H[..., ::-1, ::-1]
                )
        ret = tf.reshape(
            H, (-1, 1, len(self.outs[0].spins), len(self.outs[1].spins))
        )
        return ret

    def get_factor_angle_helicity_amp(self, data, data_p, **kwargs):
        m_dep = self.get_angle_ls_amp(data, data_p, **kwargs)  # (n,l)
        cg_trans = tf.cast(self.get_cg_matrix(), m_dep.dtype)
        n_ls = len(self.get_ls_list())
        m_dep = tf.reshape(m_dep, (-1, n_ls, 1, 1))
        cg_trans = tf.reshape(
            cg_trans, (n_ls, len(self.outs[0].spins), len(self.outs[1].spins))
        )
        # H = tf.reduce_sum(m_dep * cg_trans, axis=1)
        H = m_dep * cg_trans  # (n, n_ls, h1, h2)
        # print(n_ls, cg_trans, self, m_dep.shape) # )data_p)
        if self.allow_cc:
            all_data = kwargs.get("all_data", {})
            charge = all_data.get("charge_conjugation", None)
            if charge is not None:
                H = tf.where(
                    charge[..., None, None] > 0, H, H[..., ::-1, ::-1]
                )
        ret = tf.reshape(
            H,
            (
                -1,
                H.shape[-3],
                1,
                len(self.outs[0].spins),
                len(self.outs[1].spins),
            ),
        )
        return ret

    def get_g_ls(self):
        gls = self.g_ls()
        if self.ls_index is None:
            ret = tf.stack(gls)
        else:
            ret = tf.stack([gls[k] for k in self.ls_index])
        if self.mask_factor:
            return tf.ones_like(ret)
        return ret

    def get_ls_amp_org(self, data, data_p, **kwargs):
        g_ls = self.get_g_ls()
        # print(g_ls)
        q0 = self.get_relative_momentum(data_p, False)
        data["|q0|"] = q0
        if "|q|" in data:
            q = data["|q|"]
        else:
            q = self.get_relative_momentum(data_p, True)
            data["|q|"] = q
        if self.has_barrier_factor:
            bf = self.get_barrier_factor(data_p[self.core]["m"], q, q0, self.d)
            mag = g_ls
            m_dep = mag * tf.cast(bf, mag.dtype)
        else:
            m_dep = g_ls
        return m_dep

    def get_ls_amp(self, data, data_p, **kwargs):
        g_ls = self.get_g_ls()
        q0 = self.get_relative_momentum2(data_p, False)
        data["|q0|2"] = q0
        if "|q|2" in data:
            q = data["|q|2"]
        else:
            q = self.get_relative_momentum2(data_p, True)
            data["|q|2"] = q
        if self.has_barrier_factor:
            bf = self.get_barrier_factor2(
                data_p[self.core]["m"], q, q0, self.d
            )
            mag = g_ls
            bf = to_complex(bf)
            m_dep = mag * tf.cast(bf, mag.dtype)
        else:
            m_dep = g_ls
        return m_dep

    def get_angle_g_ls(self):
        gls = tf.ones_like(self.g_ls())
        # [complex(1.0, 0.0) for i in range(len(self.g_ls()))]
        if self.ls_index is None:
            return gls  # tf.stack(gls)
        return tf.stack([gls[k] for k in self.ls_index])

    def get_angle_ls_amp(self, data, data_p, **kwargs):
        g_ls = self.get_angle_g_ls()
        return g_ls

    def get_barrier_factor(self, mass, q, q0, d):
        ls = self.get_l_list()
        ret = []
        for l in ls:
            if self.has_bprime:
                tmp = q**l * tf.cast(Bprime(l, q, q0, d), dtype=q.dtype)
            else:
                tmp = q**l
            # tmp = tf.where(q > 0, tmp, tf.zeros_like(tmp))
            ret.append(tf.reshape(tmp, (-1, 1)))
        ret = tf.concat(ret, axis=-1)
        mass_dep = self.get_barrier_factor_mass(mass)
        return ret * mass_dep

    def get_barrier_factor2(self, mass, q2, q02, d):
        ls = self.get_l_list()
        ret = []
        for l in ls:
            if self.has_bprime:
                bp = Bprime_q2(l, q2, q02, d)
                tmp = q2 ** (l / 2) * tf.cast(bp, dtype=q2.dtype)
                if self.barrier_factor_norm:
                    tmp = tmp / tf.cast(tf.abs(q02), tmp.dtype) ** (l / 2)
            else:
                tmp = q2 ** (l / 2)
            # tmp = tf.where(q > 0, tmp, tf.zeros_like(tmp))
            ret.append(tf.reshape(tmp, (-1, 1)))
        ret = tf.concat(ret, axis=-1)
        mass_dep = self.get_barrier_factor_mass(mass)
        return ret * mass_dep

    def get_barrier_factor_mass(self, mass):
        if not self.barrier_factor_mass:
            return 1.0
        ls = tf.convert_to_tensor(self.get_l_list(), dtype=mass.dtype)
        m_dep = 1.0 / tf.pow(tf.expand_dims(mass, -1), ls)
        return m_dep

    def get_amp(self, data, data_p, **kwargs):
        a = self.core
        b = self.outs[0]
        c = self.outs[1]
        ang = data[b]["ang"]
        D_conj = get_D_matrix_lambda(ang, a.J, a.spins, b.spins, c.spins)
        H = self.get_helicity_amp(data, data_p, **kwargs)
        H = tf.reshape(
            H, (-1, 1, len(self.outs[0].spins), len(self.outs[1].spins))
        )
        H = tf.cast(H, dtype=D_conj.dtype)
        ret = H * tf.stop_gradient(D_conj)
        # print(self, H, D_conj)
        # exit()
        if self.aligned:
            for j, particle in enumerate(self.outs):
                if particle.J != 0 and "aligned_angle" in data[particle]:
                    ang = data[particle].get("aligned_angle", None)
                    if ang is None:
                        continue
                    dt = get_D_matrix_lambda(
                        ang, particle.J, particle.spins, particle.spins
                    )
                    dt_shape = [-1, 1, 1, 1, 1]
                    dt_shape[j + 2] = len(particle.spins)
                    dt_shape[j + 3] = len(particle.spins)
                    dt = tf.reshape(dt, dt_shape)
                    D_shape = [-1, len(a.spins), len(b.spins), len(c.spins)]
                    D_shape.insert(j + 3, 2)
                    D_shape[j + 3] = 1
                    ret = tf.reshape(ret, D_shape)
                    ret = dt * ret
                    ret = tf.reduce_sum(ret, axis=j + 2)
        return ret

    def get_angle_amp(self, data, data_p, **kwargs):
        a = self.core
        b = self.outs[0]
        c = self.outs[1]
        ang = data[b]["ang"]
        D_conj = get_D_matrix_lambda(ang, a.J, a.spins, b.spins, c.spins)
        H = self.get_angle_helicity_amp(data, data_p, **kwargs)
        H = tf.reshape(
            H, (-1, 1, len(self.outs[0].spins), len(self.outs[1].spins))
        )
        H = tf.cast(H, dtype=D_conj.dtype)
        ret = H * tf.stop_gradient(D_conj)
        # print(self, H, D_conj)
        # exit()
        if self.aligned:
            for j, particle in enumerate(self.outs):
                if particle.J != 0 and "aligned_angle" in data[particle]:
                    ang = data[particle].get("aligned_angle", None)
                    if ang is None:
                        continue
                    dt = get_D_matrix_lambda(
                        ang, particle.J, particle.spins, particle.spins
                    )
                    dt_shape = [-1, 1, 1, 1, 1]
                    dt_shape[j + 2] = len(particle.spins)
                    dt_shape[j + 3] = len(particle.spins)
                    dt = tf.reshape(dt, dt_shape)
                    D_shape = [-1, len(a.spins), len(b.spins), len(c.spins)]
                    D_shape.insert(j + 3, 2)
                    D_shape[j + 3] = 1
                    ret = tf.reshape(ret, D_shape)
                    ret = dt * ret
                    ret = tf.reduce_sum(ret, axis=j + 2)
        return ret

    def get_factor_angle_amp(self, data, data_p, **kwargs):
        a = self.core
        b = self.outs[0]
        c = self.outs[1]
        ang = data[b]["ang"]
        D_conj = get_D_matrix_lambda(ang, a.J, a.spins, b.spins, c.spins)
        H = self.get_factor_angle_helicity_amp(data, data_p, **kwargs)
        H = tf.cast(H, dtype=D_conj.dtype)
        D_conj = tf.reshape(D_conj, (-1, 1, *D_conj.shape[1:]))
        ret = H * tf.stop_gradient(D_conj)
        # print(self, H, D_conj)
        # exit()
        if self.aligned:
            raise NotImplemented
        return ret

    def get_m_dep(self, data, data_p, **kwargs):
        return self.get_ls_amp(data, data_p, **kwargs)

    def get_ls_list(self):
        """get possible ls for decay, with l_list filter possible l"""
        ls_list = super(HelicityDecay, self).get_ls_list()
        if self.ls_list is not None:
            return self.ls_list
        if self.l_list is None:
            return ls_list
        ret = []
        for l, s in ls_list:
            if l in self.l_list:
                ret.append((l, s))
        return tuple(ret)


@regist_decay("default", 3)
@regist_decay("AngSam3", 3)
class AngSam3Decay(AmpDecay, AmpBase):
    def init_params(self):
        a = self.core.J
        self.gi = self.add_var(
            "G_mu", is_complex=True, shape=(_spin_int(2 * a + 1),)
        )
        try:
            self.gi.set_fix_idx(fix_idx=0, fix_vals=(1.0, 0.0))
        except Exception as e:
            print(e)

    def get_amp(self, data, data_extra=None, **kwargs):
        a = self.core
        b = self.outs[0]
        c = self.outs[1]
        d = self.outs[2]
        gi = tf.stack(self.gi())
        ang = data["ang"]
        D_conj = get_D_matrix_lambda(
            ang, a.J, a.spins, tuple(_spin_range(-a.J, a.J))
        )
        ret = tf.cast(gi, D_conj.dtype) * D_conj
        ret = tf.reduce_sum(ret, axis=-1)
        ret = tf.reshape(ret, (-1, len(a.spins), 1, 1, 1))
        ret = tf.tile(ret, [1, 1, len(b.spins), len(c.spins), len(d.spins)])
        return ret


class AmpDecayChain(BaseDecayChain, AmpBase):
    def __init__(self, *args, is_cp=False, **kwargs):
        self.is_cp = is_cp
        super(AmpDecayChain, self).__init__(*args, **kwargs)
        self.aligned = True
        self.need_amp_particle = True
        self.mask_factor = False


@register_decay_chain("default")
class DecayChain(AmpDecayChain):
    """A list of Decay as a chain decay"""

    def init_params(self, name=""):
        self.total = self.add_var(
            name + "total", is_complex=True, is_cp=self.is_cp, shape=[1]
        )
        # self.total = self.add_var(name + "total", is_complex=True, shape=[1])

    def get_factor_variable(self):
        a = []
        for i in self:
            tmp = i.get_factor_variable()
            if tmp:
                a.append(tmp)
        for j in self.inner:
            tmp = j.get_factor_variable()
            if tmp:
                a.append(tmp)
        return [tuple([self.total] + a)]

    def get_factor(self):
        decay_factor = [i.get_factor() for i in self]
        particle_factor = [i.get_factor() for i in self.inner]
        all_factor = particle_factor + decay_factor
        all_factor = [i for i in all_factor if i is not None]
        all_factor = all_factor
        ret = self.get_amp_total()
        for i in all_factor:
            ret = tf.expand_dims(ret, axis=-1) * tf.cast(i, ret.dtype)
        ret = tf.reshape(ret, (-1,))
        return ret

    def get_amp_total(self, charge=1):
        if self.mask_factor:
            return tf.ones_like(tf.stack(self.total(charge)))
        return tf.stack(self.total(charge))

    def product_gls(self):
        ret = self.get_all_factor()
        return tf.reduce_prod(ret)

    def get_all_factor(self):
        ret = [self.get_amp_total()]
        for i in self:
            ret.append(i.get_g_ls())
        return ret

    def get_cp_amp_total(self, charge=1):
        if not self.is_cp:
            return self.get_amp_total()
        total_pos = self.get_amp_total(1)
        total_neg = self.get_amp_total(-1)
        # print("total_pos", total_pos)
        # print("total_neg", total_neg)
        charge_cond = charge > 0
        # print("charge", charge)
        total = tf.where(charge_cond, total_pos, total_neg)
        return total

    def get_amp(self, data_c, data_p, all_data=None, base_map=None):
        base_map = self.get_base_map(base_map)
        iter_idx = ["..."]
        amp_d = []
        indices = []
        final_indices = "".join(iter_idx + self.amp_index(base_map))
        for i in self:
            indices.append(i.amp_index(base_map))
            amp_d.append(i.get_amp(data_c[i], data_p, all_data=all_data))

        if self.need_amp_particle:
            rs = self.get_amp_particle(data_p, data_c, all_data=all_data)

            total = self.get_cp_amp_total(
                charge=all_data.get("charge_conjugation", 1)
            )
            if rs is not None:
                total = total * tf.cast(rs, total.dtype)
            # print(total)*self.get_amp_total()
            amp_d.append(total)
            indices.append([])

        if self.aligned:
            for i in self:
                for j in i.outs:
                    if j.J != 0 and "aligned_angle" in data_c[i][j]:
                        ang = data_c[i][j]["aligned_angle"]
                        dt = get_D_matrix_lambda(ang, j.J, j.spins, j.spins)
                        amp_d.append(tf.stop_gradient(dt))
                        idx = [base_map[j], base_map[j].upper()]
                        indices.append(idx)
                        final_indices = final_indices.replace(*idx)
        idxs = []
        for i in indices:
            tmp = "".join(iter_idx + i)
            idxs.append(tmp)
        idx = ",".join(idxs)
        idx_s = "{}->{}".format(idx, final_indices)
        # ret = amp * tf.reshape(rs, [-1] + [1] * len(self.amp_shape()))
        # print(idx_s)#, amp_d)
        try:
            ret = einsum(idx_s, *amp_d)
        except:
            ret = tf.einsum(idx_s, *amp_d)
        # print(self, ret[0])
        # exit()
        # ret = einsum(idx_s, *amp_d)
        return ret

    def get_angle_amp(self, data_c, data_p, all_data=None, base_map=None):
        base_map = self.get_base_map(base_map)
        iter_idx = ["..."]
        amp_d = []
        indices = []
        final_indices = "".join(iter_idx + self.amp_index(base_map))
        for i in self:
            indices.append(i.amp_index(base_map))
            amp_d.append(i.get_angle_amp(data_c[i], data_p, all_data=all_data))

        if self.aligned:
            for i in self:
                for j in i.outs:
                    if j.J != 0 and "aligned_angle" in data_c[i][j]:
                        ang = data_c[i][j]["aligned_angle"]
                        dt = get_D_matrix_lambda(ang, j.J, j.spins, j.spins)
                        amp_d.append(tf.stop_gradient(dt))
                        idx = [base_map[j], base_map[j].upper()]
                        indices.append(idx)
                        final_indices = final_indices.replace(*idx)
        idxs = []
        for i in indices:
            tmp = "".join(iter_idx + i)
            idxs.append(tmp)
        idx = ",".join(idxs)
        idx_s = "{}->{}".format(idx, final_indices)
        # ret = amp * tf.reshape(rs, [-1] + [1] * len(self.amp_shape()))
        # print(idx_s)#, amp_d)
        ret = tf.einsum(idx_s, *amp_d)
        # print(self, ret[0])
        # exit()
        # ret = einsum(idx_s, *amp_d)
        return ret

    def get_factor_angle_amp(
        self, data_c, data_p, all_data=None, base_map=None
    ):
        base_map = self.get_base_map(base_map)
        iter_idx = ["..."]
        amp_d = []
        indices = []
        next_map = "zyxwvutsr"
        used_idx = ""
        final_indices = self.amp_index(base_map)
        for i in self:
            tmp_idx = i.amp_index(base_map)
            tmp_idx = [next_map[0], *tmp_idx]
            indices.append(tmp_idx)
            used_idx += next_map[0]
            amp_d.append(
                i.get_factor_angle_amp(data_c[i], data_p, all_data=all_data)
            )
            next_map = next_map[1:]
        final_indices = "".join(iter_idx + list(used_idx) + final_indices)

        if self.aligned:
            for i in self:
                for j in i.outs:
                    if j.J != 0 and "aligned_angle" in data_c[i][j]:
                        ang = data_c[i][j]["aligned_angle"]
                        dt = get_D_matrix_lambda(ang, j.J, j.spins, j.spins)
                        amp_d.append(tf.stop_gradient(dt))
                        idx = [base_map[j], base_map[j].upper()]
                        indices.append(idx)
                        final_indices = final_indices.replace(*idx)
        idxs = []
        for i in indices:
            tmp = "".join(iter_idx + i)
            idxs.append(tmp)
        idx = ",".join(idxs)
        idx_s = "{}->{}".format(idx, final_indices)
        # ret = amp * tf.reshape(rs, [-1] + [1] * len(self.amp_shape()))
        print(idx_s)  # , amp_d)
        ret = tf.einsum(idx_s, *amp_d)
        # print(self, ret[0])
        # exit()
        # ret = einsum(idx_s, *amp_d)
        return ret

    def get_m_dep(self, data_c, data_p, all_data=None, base_map=None):
        base_map = self.get_base_map(base_map)
        iter_idx = ["..."]
        amp_d = []
        indices = []
        final_indices = "".join(iter_idx + self.amp_index(base_map))
        for i in self:
            indices.append(i.amp_index(base_map))
            amp_d.append(i.get_m_dep(data_c[i], data_p, all_data=all_data))

        if self.need_amp_particle:
            rs = self.get_amp_particle(data_p, data_c, all_data=all_data)
            total = self.get_cp_amp_total(
                all_data.get("charge_conjugation", 1)
            )
            # print("total_pos", total_pos)
            # print("total_neg", total_neg)
            if rs is not None:
                total = total * tf.cast(rs, total.dtype)
                # print("charge", charge)
            # print(total)
            # print(total)*self.get_amp_total()
            amp_d.append(total)
        return amp_d

    def get_amp_particle(self, data_p, data_c, all_data=None):
        amp_p = []
        if not self.inner:
            return 1.0
        for i in self.inner:
            if len(i.decay) >= 1:
                decay_i = i.decay[0]
                found = False
                for j in i.decay:
                    if j in self:
                        decay_i = j
                        found = True
                        break
                if not found:
                    raise IndexError(
                        "not found {} decay in {}".format(i, self)
                    )
                data_c_i = data_c[decay_i]
                if "|q|" not in data_c_i:
                    data_c_i["|q|"] = decay_i.get_relative_momentum(
                        data_p, True
                    )
                if "|q0|" not in data_c_i:
                    data_c_i["|q0|"] = decay_i.get_relative_momentum(
                        data_p, False
                    )
                if "|q|2" not in data_c_i:
                    data_c_i["|q|2"] = decay_i.get_relative_momentum2(
                        data_p, True
                    )
                if "|q0|2" not in data_c_i:
                    data_c_i["|q0|2"] = decay_i.get_relative_momentum2(
                        data_p, False
                    )
                amp_p.append(i.get_amp(data_p[i], data_c_i, all_data=all_data))
            else:
                amp_p.append(i.get_amp(data_p[i], all_data=all_data))
        rs = 1.0
        for i in amp_p:
            rs = rs * i
        # tf.reduce_prod(amp_p, axis=0)
        return rs

    def amp_shape(self):
        ret = [len(self.top.spins)]
        for i in self.outs:
            ret.append(len(i.spins))
        return tuple(ret)

    # @simple_cache_fun
    def amp_index(self, base_map=None):
        if base_map is None:
            base_map = self.get_base_map()
        ret = [base_map[self.top]]
        for i in self.outs:
            ret.append(base_map[i])
        return ret

    def get_base_map(self, base_map=None):
        gen = index_generator(base_map)
        if base_map is None:
            base_map = {}
        ret = base_map.copy()
        if self.top not in base_map:
            ret[self.top] = next(gen)
        for i in self.outs:
            if i not in base_map:
                ret[i] = next(gen)
        for i in self.inner:
            if i not in ret:
                ret[i] = next(gen)
        return ret


class DecayGroup(BaseDecayGroup, AmpBase):
    """A Group of Decay Chains with the same final particles."""

    def __init__(self, chains):
        self.chains_idx = list(range(len(chains)))
        first_chain = chains[0]
        if not isinstance(first_chain, DecayChain):
            chains = [DecayChain(i) for i in chains]
        super(DecayGroup, self).__init__(chains)
        self.not_full = False
        self.polarization = getattr(self.top, "polarization", "none")
        # self.init_params()

    def init_params(self, name=""):
        for i in self.resonances:
            i.init_params()
        inited_set = set()
        for i in self:
            i.init_params(name)
            for j in i:
                if j not in inited_set:
                    j.init_params()
                    inited_set.add(j)
        if self.polarization == "vector":
            print("add polarization vector")
            if self.top.J == 0.5:
                self.polarization_vector = [
                    self.top.add_var("polarization_px"),
                    self.top.add_var("polarization_py"),
                    self.top.add_var("polarization_pz"),
                ]

    def get_factor_variable(self):
        ret = []
        for i in self:
            ret += i.get_factor_variable()
        return ret

    def get_factor(self):
        ret = []
        for i in self:
            ret += i.get_factor()
        return ret

    def get_amp(self, data):
        """
        calculate the amplitude as complex number
        """
        data_particle = data["particle"]
        data_decay = data["decay"]

        used_chains = tuple([self.chains[i] for i in self.chains_idx])
        chain_maps = self.get_chains_map(used_chains)
        base_map = self.get_base_map()
        ret = []
        for chains in chain_maps:
            for decay_chain in chains:
                chain_topo = decay_chain.standard_topology()
                found = False
                for i in data_decay.keys():
                    if i == chain_topo:
                        data_decay_i = data_decay[i]
                        found = True
                        break
                if not found:
                    raise KeyError("not found {}".format(chain_topo))
                data_c = rename_data_dict(data_decay_i, chains[decay_chain])
                data_p = rename_data_dict(data_particle, chains[decay_chain])
                # print("$$$$$",data_c)
                # print("$$$$$",data_p)
                amp = decay_chain.get_amp(
                    data_c, data_p, base_map=base_map, all_data=data
                )
                ret.append(amp)
                # print(decay_chain, amp[:10])
        ret = tf.reduce_sum(ret, axis=0)
        return ret

    def get_m_dep(self, data):
        """get mass dependent items"""
        data_particle = data["particle"]
        data_decay = data["decay"]

        used_chains = tuple([self.chains[i] for i in self.chains_idx])
        chain_maps = self.get_chains_map(used_chains)
        base_map = self.get_base_map()
        ret = []
        for decay_chain in used_chains:
            for chains in chain_maps:
                if str(decay_chain) in [str(i) for i in chains]:
                    maps = chains[decay_chain]
                    break
            chain_topo = decay_chain.standard_topology()
            found = False
            for i in data_decay.keys():
                if i == chain_topo:
                    data_decay_i = data_decay[i]
                    found = True
                    break
            if not found:
                raise KeyError("not found {}".format(chain_topo))
            data_c = rename_data_dict(data_decay_i, maps)
            data_p = rename_data_dict(data_particle, maps)
            # print("$$$$$",data_c)
            # print("$$$$$",data_p)
            amp = decay_chain.get_m_dep(
                data_c, data_p, base_map=base_map, all_data=data
            )
            ret.append(amp)
        # ret = tf.reduce_sum(ret, axis=0)
        return ret

    def get_angle_amp(self, data):
        data_particle = data["particle"]
        data_decay = data["decay"]

        used_chains = tuple([self.chains[i] for i in self.chains_idx])
        chain_maps = self.get_chains_map(used_chains)
        base_map = self.get_base_map()
        ret = []
        for decay_chain in used_chains:
            for chains in chain_maps:
                if str(decay_chain) in [str(i) for i in chains]:
                    maps = chains[decay_chain]
                    break
            chain_topo = decay_chain.standard_topology()
            found = False
            for i in data_decay.keys():
                if i == chain_topo:
                    data_decay_i = data_decay[i]
                    found = True
                    break
            if not found:
                raise KeyError("not found {}".format(chain_topo))
            data_c = rename_data_dict(data_decay_i, maps)
            data_p = rename_data_dict(data_particle, maps)
            amp = decay_chain.get_angle_amp(
                data_c, data_p, base_map=base_map, all_data=data
            )
            ret.append(amp)
        # ret = tf.reduce_sum(ret, axis=0)
        return amp

    def get_factor_angle_amp(self, data):
        data_particle = data["particle"]
        data_decay = data["decay"]

        used_chains = tuple([self.chains[i] for i in self.chains_idx])
        chain_maps = self.get_chains_map(used_chains)
        base_map = self.get_base_map()
        ret = []
        for decay_chain in used_chains:
            for chains in chain_maps:
                if str(decay_chain) in [str(i) for i in chains]:
                    maps = chains[decay_chain]
                    break
            chain_topo = decay_chain.standard_topology()
            found = False
            for i in data_decay.keys():
                if i == chain_topo:
                    data_decay_i = data_decay[i]
                    found = True
                    break
            if not found:
                raise KeyError("not found {}".format(chain_topo))
            data_c = rename_data_dict(data_decay_i, maps)
            data_p = rename_data_dict(data_particle, maps)
            amp = decay_chain.get_factor_angle_amp(
                data_c, data_p, base_map=base_map, all_data=data
            )
            ret.append(amp)
        # ret = tf.reduce_sum(ret, axis=0)
        return amp

    @functools.lru_cache()
    def get_swap_factor(self, key):
        factor = 1.0
        used = []
        for i, j in zip(self.identical_particles, key[1]):
            p = self.get_particle(i[0])
            if int(p.J * 2) % 2 == 0:
                continue
            for m, n in zip(i, j):
                if (m, n) in used or (n, m) in used:
                    continue
                used.append((m, n))
                if m != n:
                    factor *= -1.0
        return factor

    @functools.lru_cache()
    def get_id_swap_transpose(self, key, n):
        _, change = key
        # print(key)
        old_order = [str(i) for i in self.outs]
        trans = []
        for i, j in zip(self.identical_particles, change):
            for k, l in zip(i, j):
                trans.append((k, l))
        trans = tuple(trans)
        return self.get_swap_transpose(trans, n)

    @functools.lru_cache()
    def get_swap_transpose(self, trans, n):
        trans = dict(trans)
        # print(trans)
        tmp = {v: k for k, v in trans.items()}
        tmp.update(trans)
        trans = tmp
        # print(trans)
        old_order = [str(i) for i in self.outs]
        new_order = []
        for i in old_order:
            new_order.append(trans.get(i, i))
        index_map = {k: i for i, k in enumerate(new_order)}
        trans_order = [index_map[str(i)] for i in self.outs]
        diff = n - len(trans_order)
        return [i for i in range(diff)] + [i + diff for i in trans_order]

    def get_amp2(self, data):
        amp = self.get_amp(data)
        id_swap = data.get("id_swap", {})
        for k, v in id_swap.items():
            new_data = {**data, **v}
            factor = self.get_swap_factor(k)
            amp_swap = factor * self.get_amp(new_data)
            # print(k, amp, amp_swap)
            swap_index = self.get_id_swap_transpose(k, len(amp_swap.shape))
            # print(swap_index)
            amp_swap = tf.transpose(amp_swap, swap_index)
            amp = amp + amp_swap
        return amp

    def get_amp3(self, data):
        amp = self.get_amp2(data)
        if "cp_swap" in data:
            amp_swap = self.get_amp2(data["cp_swap"])
            cg = cp_charge_group(
                [str(i) for i in self.outs],
                self.identical_particles,
                self.cp_particles,
            )
            name_map = {str(i): i for i in self.outs}
            frac = 1.0
            same_particle = []
            change = []
            for a, b in cg:
                for i, j in zip(a, b):
                    if i == j:
                        same_particle.append(i)
                        frac = frac * getattr(name_map[i], "C", -1)
                    else:
                        change.append((i, j))
            transpose = self.get_swap_transpose(
                tuple(change), len(amp_swap.shape)
            )
            p_reverse = [Ellipsis] + [
                slice(None, None, -1) for i in range(len(amp_swap.shape) - 1)
            ]

            amp = (
                amp
                + tf.transpose(amp_swap, transpose).__getitem__(p_reverse)
                * frac
            )
        return amp

    def sum_amp(self, data, cached=True):
        """
        calculat the amplitude modular square
        """
        if not cached:
            data = simple_deepcopy(data)
        if self.polarization != "none":
            return self.sum_amp_polarization(data)
        amp = self.get_amp3(data)
        amp2s = tf.math.real(amp * tf.math.conj(amp))
        idx = list(range(1, len(amp2s.shape)))
        sum_A = tf.reduce_sum(amp2s, idx)
        return sum_A

    def sum_amp_polarization(self, data):
        """
        sum amplitude suqare with density _get_cg_matrix

        .. math::
            P = \\sum_{m, m', \\cdots } A_{m, \\cdots}  \\rho_{m, m'} A^{*}_{m', \\cdots}

        """

        amp = self.get_amp3(data)
        amp = tf.reshape(
            amp, (amp.shape[0], amp.shape[1], -1)
        )  # (i, la, lb lc ld ...)
        na, nl = amp.shape[1], amp.shape[2]
        rho = self.get_density_matrix()
        amp = tf.reshape(amp, (-1, na, 1, nl))
        amp_c = tf.reshape(
            tf.math.conj(amp), (-1, na, nl)
        )  # (i, la, lb lc ld ...)
        sum_A = (
            tf.reduce_sum(amp * tf.reshape(rho, (na, na, 1)), axis=1) * amp_c
        )
        return tf.reduce_sum(tf.math.real(sum_A), axis=[1, 2])

    def get_density_matrix(self):
        if self.polarization == "vector":
            px, py, pz = [i() for i in self.polarization_vector]
            zeros = tf.zeros_like(px)
            ones = tf.ones_like(px)
            rho00 = tf.complex(ones + pz, zeros)
            rho11 = tf.complex(ones - pz, zeros)
            rho01 = tf.complex(px, -py)
            rho10 = tf.complex(px, py)
            ret = 0.5 * tf.stack([[rho00, rho01], [rho10, rho11]])
            # print(ret)
            return ret
        raise NotImplementedError

    # @simple_cache_fun
    def amp_index(self, gen=None, base_map=None):
        if base_map is None:
            base_map = self.get_base_map()
        ret = [base_map[self.top]]
        for i in self.outs:
            ret.append(base_map[i])
        return ret

    def get_base_map(self, gen=None, base_map=None):
        if gen is None:
            gen = index_generator(base_map)
        if base_map is None:
            base_map = {self.top: next(gen)}
        for i in self.outs:
            base_map[i] = next(gen)
        return base_map

    def get_res_map(self):
        res_map = {}
        for i, decay in enumerate(self.chains):
            for j in decay.inner:
                if j not in res_map:
                    res_map[j] = []
                res_map[j].append(i)
        return res_map

    def set_used_res(self, res, only=False):
        if not isinstance(res, (list, tuple)):
            res = [res]
        res_set = set()
        idx_chains = []
        for i in res:
            if isinstance(i, str):
                res_set.add(BaseParticle(i))
            elif isinstance(i, BaseParticle):
                res_set.add(i)
            elif isinstance(i, int):
                idx_chains.append(i)
            else:
                raise TypeError(
                    "type({}) = {} not a Particle".format(i, type(i))
                )
        if not only:
            used_res = set()
            for i in res_set:
                for j, c in enumerate(self.chains):
                    if i in c.inner:
                        used_res.add(j)
            self.set_used_chains(list(used_res))
        else:
            unused_res = set(self.resonances) - res_set
            unused_decay = set()
            res_map = self.get_res_map()
            for i in unused_res:
                for j in res_map[i]:
                    unused_decay.add(j)
            used_decay = []
            for i, _ in enumerate(self.chains):
                if i not in unused_decay:
                    used_decay.append(i)
            self.set_used_chains(used_decay)
        self.add_used_chains(idx_chains)

    def add_used_chains(self, used_chains):
        for i in used_chains:
            assert isinstance(i, int), "not index of chains"
            if i in self.chains_idx:
                continue
            else:
                self.chains_idx.append(i)

    def set_used_chains(self, used_chains):
        self.chains_idx = list(used_chains)
        if len(self.chains_idx) != len(self.chains):
            self.not_full = True
        else:
            self.not_full = False

    def partial_weight(self, data, combine=None):
        chains = list(self.chains)
        if combine is None:
            combine = [[i] for i in range(len(chains))]
        o_used_chains = self.chains_idx
        weights = []
        for i in combine:
            self.set_used_res(i)
            weight = self.sum_amp(data)
            weights.append(weight)
        self.set_used_chains(o_used_chains)
        return weights

    def chains_particle(self):
        ret = []
        for i in self:
            ret.append(tuple(i.inner))
        return ret

    def partial_weight_interference(self, data):
        chains = list(self.chains)
        combine = combinations(range(len(chains)), 2)
        o_used_chains = self.chains_idx
        weights = {}
        for i in combine:
            self.set_used_chains(i)
            weight = self.sum_amp(data)
            weights[i] = weight
        self.set_used_chains(o_used_chains)
        return weights

    def generate_phasespace(self, num=100000):
        def get_mass(i):
            mass = i.get_mass()
            if mass is None:
                raise Exception("mass is required for particle {}".format(i))
            return mass

        top_mass = get_mass(self.top)
        final_mass = [get_mass(i) for i in self.outs]
        from tf_pwa.phasespace import PhaseSpaceGenerator

        a = PhaseSpaceGenerator(top_mass, final_mass)
        data = a.generate(num)
        return dict(zip(self.outs, data))


def index_generator(base_map=None):
    indices = "abcdefghjklmnopqrstuvwxyz"
    if base_map is not None:
        for i in base_map:
            indices = indices.replace(base_map[i], "")
    for i in indices:
        yield i


def rename_data_dict(data, idx_map):
    if isinstance(data, dict):
        return {
            idx_map.get(k, k): rename_data_dict(v, idx_map)
            for k, v in data.items()
        }
    if isinstance(data, tuple):
        return tuple([rename_data_dict(i, idx_map) for i in data])
    if isinstance(data, list):
        return [rename_data_dict(i, idx_map) for i in data]
    return data


def value_and_grad(f, var):
    with tf.GradientTape() as tape:
        s = f(var)
    g = tape.gradient(s, var)
    return s, g


def load_decfile_particle(fname):
    with open(fname) as f:
        dec = load_dec_file(f)
    dec = list(dec)
    particles = {}

    def get_particles(name):
        if name not in particles:
            a = get_particle(name)
            particles[name] = a
        return particles[name]

    decay = []
    for i in dec:
        cmd, var = i
        if cmd == "Particle":
            a = get_particles(var["name"])
            setattr(a, "params", var["params"])
        if cmd == "Decay":
            for j in var["final"]:
                outs = [get_particles(k) for k in j["outs"]]
                de = Decay(get_particles(var["name"]), outs)
                for k in j:
                    if k != "outs":
                        setattr(de, k, j[k])
                decay.append(de)
        if cmd == "RUNNINGWIDTH":
            pa = get_particles(var[0])
            setattr(pa, "running_width", True)
    top, inner, outs = split_particle_type(decay)
    return top, inner, outs


regist_config(DEFAULT_DECAY, (HelicityDecay, {}))
