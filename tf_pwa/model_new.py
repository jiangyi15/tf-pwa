import numpy as np
from .data import data_shape, split_generator, data_merge, data_split
from .tensorflow_wrapper import tf
from .utils import time_print
from .config import get_config


def loop_generator(var):
    while True:
        yield var


def sum_gradient(f, data, var, weight=1.0, trans=tf.identity, args=(), kwargs=None):
    kwargs = kwargs if kwargs is not None else {}
    if isinstance(weight, float):
        weight = loop_generator(weight)
    ys = []
    gs = []
    for data_i, weight_i in zip(data, weight):
        with tf.GradientTape() as tape:
            part_y = trans(f(data_i, *args, **kwargs))
            y_i = tf.reduce_sum(tf.cast(weight_i, part_y.dtype) * part_y)
        g_i = tape.gradient(y_i, var)
        ys.append(y_i)
        gs.append(g_i)
    nll = sum(ys)
    g = list(map(sum, zip(*gs)))
    return nll, g


def sum_hessian(f, data, var, weight=1.0, trans=tf.identity, args=(), kwargs=None):
    kwargs = kwargs if kwargs is not None else {}
    if isinstance(weight, float):
        weight = loop_generator(weight)
    y_s = []
    g_s = []
    h_s = []
    for data_i, weight_i in zip(data, weight):
        with tf.GradientTape(persistent=True) as tape0:
            with tf.GradientTape() as tape:
                part_y = trans(f(data_i, *args, **kwargs))
                y_i = tf.reduce_sum(tf.cast(weight_i, part_y.dtype) * part_y)
            g_i = tape.gradient(y_i, var)
        h_s_i = []
        for gi in g_i:
            h_s_i.append(tape0.gradient(gi, var, unconnected_gradients="zero"))  # 2nd order derivative
        del tape0
        y_s.append(y_i)
        g_s.append(g_i)
        h_s.append(h_s_i)
    nll = tf.reduce_sum(y_s)
    g = tf.reduce_sum(g_s, axis=0)
    h = tf.reduce_sum(h_s, axis=0)
    # h = [[sum(j) for j in zip(*i)] for i in h_s]
    return nll, g, h


class Model(object):
    def __init__(self, amp, w_bkg=1.0):
        self.Amp = amp
        self.w_bkg = w_bkg

    def get_weight_data(self, data, weight=1.0, bg=None, alpha=True):
        has_bg = False
        if isinstance(weight, float):
            n_data = data_shape(data)
            weight = tf.convert_to_tensor([weight]*n_data, dtype=get_config("dtype"))
        if bg is not None:
            n_bg = data_shape(bg)
            data = data_merge(data, bg)
            bg_weight = tf.convert_to_tensor([-self.w_bkg] * n_bg, dtype=get_config("dtype"))
            weight = tf.concat([weight, bg_weight], axis=0)
        if alpha:
            alpha = tf.reduce_sum(weight) / tf.reduce_sum(weight * weight)
            return data, alpha * weight
        return data, weight

    def nll(self, data, mcdata, weight: tf.Tensor = 1.0, batch=None, bg=None):
        r"""
        calculate negative log-likelihood

        .. math::
          -\ln L = -\sum_{x_i \in data } w_i \ln f(x_i;\theta_k) +  (\sum w_j ) \ln \sum_{x_i \in mc } f(x_i;\theta_k)

        """
        data, weight = self.get_weight_data(data, weight, bg=bg)
        sw = tf.reduce_sum(weight)
        ln_data = tf.math.log(self.Amp(data))
        int_mc = tf.math.log(tf.reduce_mean(self.Amp(mcdata)))
        nll_0 = - tf.reduce_sum(tf.cast(weight, ln_data.dtype) * ln_data)
        return nll_0 + tf.cast(sw, int_mc.dtype) * int_mc

    def nll_grad(self, data, mcdata, weight=1.0, batch=65000, bg=None):
        r"""
        calculate negative log-likelihood with gradients

        .. math::
          - \frac{\partial \ln L}{\partial \theta_k } =
            -\sum_{x_i \in data } w_i \frac{\partial}{\partial \theta_k} \ln f(x_i;\theta_k)
            + (\sum w_j ) \left( \frac{ \partial }{\partial \theta_k} \sum_{x_i \in mc} f(x_i;\theta_k) \right)
              \frac{1}{ \sum_{x_i \in mc} f(x_i;\theta_k) }

        """
        data, weight = self.get_weight_data(data, weight, bg=bg)
        n_mc = data_shape(mcdata)
        sw = tf.reduce_sum(weight)
        ln_data, g_ln_data = sum_gradient(self.Amp, split_generator(data, batch),
                                          self.Amp.trainable_variables, weight=split_generator(weight, batch),
                                          trans=tf.math.log)
        int_mc, g_int_mc = sum_gradient(self.Amp, split_generator(mcdata, batch),
                                        self.Amp.trainable_variables, weight=1/n_mc)

        sw = tf.cast(sw, ln_data.dtype)

        g = list(map(lambda x: - x[0] + sw * x[1] / int_mc, zip(g_ln_data, g_int_mc)))
        nll = - ln_data + sw * tf.math.log(int_mc)
        return nll, g

    # @tf.function
    def nll_grad_batch(self, data, mcdata, weight, mc_weight):
        r"""
        calculate negative log-likelihood with gradients

        .. math::
          - \frac{\partial \ln L}{\partial \theta_k } =
            -\sum_{x_i \in data } w_i \frac{\partial}{\partial \theta_k} \ln f(x_i;\theta_k)
            + (\sum w_j ) \left( \frac{ \partial }{\partial \theta_k} \sum_{x_i \in mc} f(x_i;\theta_k) \right)
              \frac{1}{ \sum_{x_i \in mc} f(x_i;\theta_k) }

        """
        sw = tf.reduce_sum(weight)
        ln_data, g_ln_data = sum_gradient(self.Amp, data,
                                          self.Amp.trainable_variables, weight=weight, trans=tf.math.log)
        int_mc, g_int_mc = sum_gradient(self.Amp, mcdata,
                                        self.Amp.trainable_variables, weight=mc_weight)

        sw = tf.cast(sw, ln_data.dtype)

        g = list(map(lambda x: - x[0] + sw * x[1] / int_mc, zip(g_ln_data, g_int_mc)))
        nll = - ln_data + sw * tf.math.log(int_mc)
        return nll, g

    def nll_grad_hessian(self, data, mcdata, weight=1.0, batch=24000, bg=None):
        data, weight = self.get_weight_data(data, weight, bg=bg)
        n_mc = data_shape(mcdata)
        sw = tf.reduce_sum(weight)
        ln_data, g_ln_data, h_ln_data = sum_hessian(self.Amp, split_generator(data, batch),
                                                    self.Amp.trainable_variables, weight=split_generator(weight, batch),
                                                    trans=tf.math.log)
        int_mc, g_int_mc, h_int_mc = sum_hessian(self.Amp, split_generator(mcdata, batch),
                                                 self.Amp.trainable_variables)

        n_var = len(g_ln_data)
        nll = - ln_data + sw * tf.math.log(int_mc / n_mc)
        g = - g_ln_data + sw * g_int_mc / int_mc
        g_outer = tf.reshape(g_int_mc, (-1, 1)) * tf.reshape(g_int_mc, (1, -1))
        h = - h_ln_data - sw * g_outer + sw / int_mc * h_int_mc
        return nll, g, h

    def set_params(self, var):
        self.Amp.set_params(var)

    def get_params(self):
        self.Amp.get_params()


class FCN(object):
    def __init__(self, model, data, mcdata, bg=None, batch=65000):
        self.model = model
        self.n_call = 0
        self.n_grad = 0
        self.cached_nll = None
        data, weight = self.model.get_weight_data(data, bg=bg)
        n_mcdata = data_shape(mcdata)
        self.alpha = tf.reduce_sum(weight) / tf.reduce_sum(weight * weight)
        self.weight = weight
        self.data = data
        self.batch_data = list(split_generator(data, batch))
        self.mcdata = mcdata
        self.batch_mcdata = list(split_generator(mcdata, batch))
        self.batch = batch
        self.mc_weight = tf.convert_to_tensor([1/n_mcdata] * n_mcdata, dtype="float64")

    # @time_print
    def __call__(self, x):
        self.model.set_params(x)
        nll = self.model.nll(self.data, self.mcdata, weight=self.weight)
        self.cached_nll = nll
        self.n_call += 1
        return nll

    def grad(self, x):
        nll, g = self.nll_grad(x)
        return g

    @time_print
    def nll_grad(self, x):
        self.model.set_params(x)
        nll, g = self.model.nll_grad_batch(self.batch_data, self.batch_mcdata,
                                           weight=list(data_split(self.weight, self.batch)),
                                           mc_weight=data_split(self.mc_weight, self.batch))
        self.cached_nll = nll
        self.n_call += 1
        return nll, np.array(g)

    def nll_grad_hessian(self, x, batch=None):
        if batch is None:
            batch = self.batch
        self.model.set_params(x)
        nll, g, h = self.model.nll_grad_hessian(self.data, self.mcdata,
                                                weight=self.weight, batch=batch)
        return nll, g, h
