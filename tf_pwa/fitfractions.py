import functools

import numpy as np
import tensorflow as tf

from tf_pwa.data import LazyCall, data_split


def eval_integral(
    f, data, var, weight=None, args=(), no_grad=False, kwargs=None
):
    kwargs = {} if kwargs is None else kwargs
    weight = 1.0 if weight is None else weight
    if no_grad:
        ret = tf.reduce_sum(f(data, *args, **kwargs) * weight)
        ret_grad = np.zeros((len(var),))
    else:
        with tf.GradientTape() as tape:
            ret = tf.reduce_sum(f(data, *args, **kwargs) * weight)
        ret_grad = tape.gradient(ret, var, unconnected_gradients="zero")
        ret_grad = np.stack([i.numpy() for i in ret_grad])
    return ret.numpy(), ret_grad


class FitFractions:
    def __init__(self, amp, res):
        self.amp = amp
        self.var = amp.trainable_variables
        self.n_var = len(amp.trainable_variables)
        self.res = res
        self.cached_int = {}
        self.cached_grad = {}
        self.cached_int_total = 0.0
        self.cached_grad_total = np.zeros((self.n_var,))
        self.error_matrix = np.diag(np.zeros((self.n_var,)))
        self.init_res_table()

    def init_res_table(self):
        n = len(self.res)
        self.cached_int_total = 0.0
        self.cached_grad_total = np.zeros((self.n_var,))
        for i in range(n):
            for j in range(i, -1, -1):
                if i == j:
                    name = str(self.res[i])
                else:
                    name = (str(self.res[i]), str(self.res[j]))
                self.cached_int[name] = 0.0
                self.cached_grad[name] = np.zeros_like((self.n_var,))

    def integral(self, mcdata, *args, batch=None, no_grad=False, **kwargs):
        self.init_res_table()
        if batch is None:
            self.append_int(mcdata, *args, no_grad=no_grad, **kwargs)
        else:
            for data_i in data_split(mcdata, batch):
                self.append_int(data_i, *args, no_grad=no_grad, **kwargs)

    def append_int(self, mcdata, *args, weight=None, no_grad=False, **kwargs):
        # print(data, data_shape(data))
        if isinstance(mcdata, LazyCall):
            mcdata = mcdata.eval()
        if weight is None:
            weight = mcdata.get("weight", 1.0)
        int_mc, g_int_mc = eval_integral(
            self.amp,
            mcdata,
            var=self.var,
            weight=weight,
            args=args,
            kwargs=kwargs,
        )
        self.cached_int_total += int_mc
        self.cached_grad_total += g_int_mc
        cahced_res = self.amp.used_res
        amp_tmp = self.amp
        for i in range(len(self.res)):
            for j in range(i, -1, -1):
                if i == j:
                    name = str(self.res[i])
                    amp_tmp.set_used_res([self.res[i]])
                else:
                    name = (str(self.res[i]), str(self.res[j]))
                    amp_tmp.set_used_res([self.res[i], self.res[j]])
                int_tmp, g_int_tmp = eval_integral(
                    amp_tmp,
                    mcdata,
                    var=self.var,
                    weight=weight,
                    args=args,
                    kwargs=kwargs,
                )
                self.cached_int[name] = self.cached_int[name] + int_tmp
                self.cached_grad[name] = self.cached_grad[name] + g_int_tmp

        self.amp.set_used_res(cahced_res)

    def get_frac_grad(self, sum_diag=True):
        n = len(self.res)
        int_mc = self.cached_int_total
        g_int_mc = self.cached_grad_total
        fit_frac = {}
        g_fit_frac = {}
        for i in range(n):
            name = str(self.res[i])
            int_tmp = self.cached_int[name]
            g_int_tmp = self.cached_grad[name]
            fit_frac[name] = int_tmp / int_mc
            gij = g_int_tmp / int_mc - (int_tmp / int_mc) * g_int_mc / int_mc
            g_fit_frac[name] = gij
            for j in range(i - 1, -1, -1):
                name = (str(self.res[i]), str(self.res[j]))
                int_tmp = self.cached_int[name]
                g_int_tmp = self.cached_grad[name]
                fit_frac[name] = (
                    (int_tmp / int_mc)
                    - fit_frac[str(self.res[i])]
                    - fit_frac[str(self.res[j])]
                )
                gij = (
                    g_int_tmp / int_mc
                    - (int_tmp / int_mc) * g_int_mc / int_mc
                    - g_fit_frac[str(self.res[i])]
                    - g_fit_frac[str(self.res[j])]
                )
                # print(name,gij.tolist())
                g_fit_frac[name] = gij
        if sum_diag:
            fit_frac["sum_diag"] = sum([fit_frac[str(i)] for i in self.res])
            g_fit_frac["sum_diag"] = sum(
                [g_fit_frac[str(i)] for i in self.res]
            )
        print(fit_frac)
        return fit_frac, g_fit_frac

    def get_frac(self, error_matrix=None, sum_diag=True):
        if error_matrix is None:
            error_matrix = self.error_matrix
        fit_frac, g_fit_frac = self.get_frac_grad(sum_diag=sum_diag)
        if error_matrix is None:
            return fit_frac, {}
        fit_frac_err = {}
        for k, v in g_fit_frac.items():
            e = np.sqrt(np.dot(np.dot(error_matrix, v), v))
            fit_frac_err[k] = e
        return fit_frac, fit_frac_err

    def __iter__(self):
        return iter(self.get_frac())

    def get_frac_diag_sum(self, error_matrix=None):
        if error_matrix is None:
            error_matrix = self.error_matrix
        sd = 0.0
        sd_g = np.zeros((self.n_var,))
        for i in self.cached_int:
            if isinstance(i, str):
                sd = sd + self.cached_int[i]
                sd_g += self.cached_grad[i]
        sd_e = np.sqrt(np.dot(np.dot(error_matrix, sd_g), sd_g))
        return sd, sd_e


def nll_grad(f, var, args=(), kwargs=None, options=None):
    kwargs = kwargs if kwargs is not None else {}
    options = options if options is not None else {}

    @functools.wraps(f)
    def f_w():
        with tf.GradientTape() as tape:
            ret = f(*args, **kwargs)
        g = tape.gradient(ret, var, unconnected_gradients="zero", **options)
        return ret, g

    return f_w


def cal_fitfractions(amp, mcdata, res=None, batch=None, args=(), kwargs=None):
    r"""
    defination:

    .. math::
        FF_{i} = \frac{\int |A_i|^2 d\Omega }{ \int |\sum_{i}A_i|^2 d\Omega }
        \approx \frac{\sum |A_i|^2 }{\sum|\sum_{i} A_{i}|^2}

    interference fitfraction:

    .. math::
        FF_{i,j} = \frac{\int 2Re(A_i A_j*) d\Omega }{ \int |\sum_{i}A_i|^2 d\Omega }
        = \frac{\int |A_i +A_j|^2  d\Omega }{ \int |\sum_{i}A_i|^2 d\Omega } - FF_{i} - FF_{j}

    gradients (for error transfer):

    .. math::
        \frac{\partial }{\partial \theta_i }\frac{f(\theta_i)}{g(\theta_i)} =
        \frac{\partial f(\theta_i)}{\partial \theta_i} \frac{1}{g(\theta_i)} -
        \frac{\partial g(\theta_i)}{\partial \theta_i} \frac{f(\theta_i)}{g^2(\theta_i)}

    """
    kwargs = kwargs if kwargs is not None else {}
    var = amp.trainable_variables
    # allvar = [i.name for i in var]
    cahced_res = amp.used_res
    if res is None:
        res = list(amp.res)
    n_res = len(res)
    fitFrac = {}
    err_fitFrac = {}
    g_fitFrac = [None] * n_res
    amp.set_used_res(res)
    weight = 1.0
    if batch is not None:
        weight = mcdata.get("weight", 1.0)
        mcdata = list(data_split(mcdata, batch))
        if not isinstance(weight, float):
            weight = list(data_split(weight, batch))
    int_mc, g_int_mc = sum_gradient(
        amp, mcdata, var=var, weight=weight, args=args, kwargs=kwargs
    )
    for i in range(n_res):
        for j in range(i, -1, -1):
            amp_tmp = amp
            if i == j:
                name = "{}".format(res[i])
                amp_tmp.set_used_res([res[i]])
            else:
                name = (str(res[i]), str(res[j]))
                amp_tmp.set_used_res([res[i], res[j]])
            int_tmp, g_int_tmp = sum_gradient(
                amp_tmp,
                mcdata,
                var=var,
                weight=weight,
                args=args,
                kwargs=kwargs,
            )
            if i == j:
                fitFrac[name] = int_tmp / int_mc
                gij = (
                    g_int_tmp / int_mc - (int_tmp / int_mc) * g_int_mc / int_mc
                )
                g_fitFrac[i] = gij
            else:
                fitFrac[name] = (
                    (int_tmp / int_mc)
                    - fitFrac["{}".format(res[i])]
                    - fitFrac["{}".format(res[j])]
                )
                gij = (
                    g_int_tmp / int_mc
                    - (int_tmp / int_mc) * g_int_mc / int_mc
                    - g_fitFrac[i]
                    - g_fitFrac[j]
                )
            # print(name,gij.tolist())
            err_fitFrac[name] = gij
    amp.set_used_res(cahced_res)
    return fitFrac, err_fitFrac


def cal_fitfractions_no_grad(
    amp, mcdata, res=None, batch=None, args=(), kwargs=None
):
    r"""
    calculate fit fractions without gradients.
    """
    kwargs = kwargs if kwargs is not None else {}
    var = amp.trainable_variables
    # allvar = [i.name for i in var]
    cahced_res = amp.used_res
    if res is None:
        res = list(amp.res)
    n_res = len(res)
    fitFrac = {}
    amp.set_used_res(res)
    weight = 1.0
    if batch is not None:
        weight = mcdata.get("weight", 1.0)
        mcdata = list(data_split(mcdata, batch))
        if not isinstance(weight, float):
            weight = list(data_split(weight, batch))
    int_mc = sum_no_gradient(
        amp, mcdata, var=var, weight=weight, args=args, kwargs=kwargs
    )
    for i in range(n_res):
        for j in range(i, -1, -1):
            amp_tmp = amp
            if i == j:
                name = "{}".format(res[i])
                amp_tmp.set_used_res([res[i]])
            else:
                name = "{}x{}".format(res[i], res[j])
                amp_tmp.set_used_res([res[i], res[j]])
            int_tmp = sum_no_gradient(
                amp_tmp,
                mcdata,
                var=var,
                weight=weight,
                args=args,
                kwargs=kwargs,
            )
            if i == j:
                fitFrac[name] = int_tmp / int_mc
            else:
                fitFrac[name] = (
                    (int_tmp / int_mc)
                    - fitFrac["{}".format(res[i])]
                    - fitFrac["{}".format(res[j])]
                )
    amp.set_used_res(cahced_res)
    return fitFrac


def sum_gradient(
    amp,
    data,
    var,
    weight=1.0,
    func=lambda x: x,
    grad=True,
    args=(),
    kwargs=None,
):
    kwargs = kwargs if kwargs is not None else {}
    # n_variables = len(var)
    if isinstance(weight, float):  # 给data乘weight
        weight = [weight] * len(data)
    nll_list = []
    g_list = []

    def f(d, w):
        amp2s = amp(d, *args, **kwargs)  # amp是振幅表达式
        l_a = func(amp2s)  # ampPDF转换成每组数据的NLL表达式
        return tf.reduce_sum(tf.cast(w, l_a.dtype) * l_a)

    for d, w in zip(data, weight):
        if grad:  # 是否提供grad表达式
            p_nll, a = nll_grad(
                f, var, args=(d, w)
            )()  # 调用上面定义的nll_grad，返回f(d,w)和f对var的导数
            g_list.append([i.numpy() for i in a])
        else:
            p_nll = f(d, w)
        nll_list.append(p_nll.numpy())
    nll = sum(nll_list)
    if grad:
        g = np.array(g_list).sum(0)
        return nll, g  # nll值和各var的导数g值
    return nll


sum_no_gradient = functools.partial(sum_gradient, grad=False)
