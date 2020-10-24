import numpy as np

import tensorflow as tf
from tf_pwa.data import data_shape
from tf_pwa.experimental import build_amp, opt_int

from .model import (Model, _loop_generator, clip_log, split_generator,
                    sum_hessian)


def sum_gradient(fs, var, weight=1.0, trans=tf.identity, args=(), kwargs=None):
    """
    NLL is the sum of trans(f(data)):math:`*`weight; gradient is the derivatives for each variable in ``var``.

    :param f: Function. The amplitude PDF.
    :param var: List of strings. Names of the trainable variables in the PDF.
    :param weight: Weight factor for each data point. It's either a real number or an array of the same shape with ``data``.
    :param trans: Function. Transformation of ``data`` before multiplied by ``weight``.
    :param kwargs: Further arguments for ``f``.
    :return: Real number NLL, list gradient
    """
    kwargs = kwargs if kwargs is not None else {}
    if isinstance(weight, float):
        weight = _loop_generator(weight)
    ys = []
    gs = []
    for f, weight_i in zip(fs, weight):
        with tf.GradientTape() as tape:
            part_y = trans(f(*args, **kwargs))
            y_i = tf.reduce_sum(tf.cast(weight_i, part_y.dtype) * part_y)
        g_i = tape.gradient(y_i, var, unconnected_gradients="zero")
        ys.append(y_i)
        gs.append(g_i)
    nll = sum(ys)
    g = list(map(sum, zip(*gs)))
    return nll, g


def sum_gradient_data2(
    f,
    var,
    data,
    cached_data,
    weight=1.0,
    trans=tf.identity,
    args=(),
    kwargs=None,
):
    """
    NLL is the sum of trans(f(data)):math:`*`weight; gradient is the derivatives for each variable in ``var``.

    :param f: Function. The amplitude PDF.
    :param var: List of strings. Names of the trainable variables in the PDF.
    :param weight: Weight factor for each data point. It's either a real number or an array of the same shape with ``data``.
    :param trans: Function. Transformation of ``data`` before multiplied by ``weight``.
    :param kwargs: Further arguments for ``f``.
    :return: Real number NLL, list gradient
    """
    kwargs = kwargs if kwargs is not None else {}
    if isinstance(weight, float):
        weight = _loop_generator(weight)
    ys = []
    gs = []
    for data_i, c_data_i, weight_i in zip(data, cached_data, weight):
        with tf.GradientTape() as tape:
            part_y = trans(f(data_i, c_data_i, *args, **kwargs))
            y_i = tf.reduce_sum(tf.cast(weight_i, part_y.dtype) * part_y)
        g_i = tape.gradient(y_i, var, unconnected_gradients="zero")
        ys.append(y_i)
        gs.append(g_i)
    nll = sum(ys)
    g = list(map(sum, zip(*gs)))
    return nll, g


class ModelCachedInt(Model):
    """
    This class implements methods to calculate NLL as well as its derivatives for an amplitude model with Cached Int.
    It may include data for both signal and background.
    Cached Int well cause wrong results when float parameters include mass or width.

    :param amp: ``AllAmplitude`` object. The amplitude model.
    :param w_bkg: Real number. The weight of background.
    """

    def __init__(self, amp, w_bkg=1.0):
        self.Amp = amp
        self.w_bkg = w_bkg
        self.vm = amp.vm
        self.cached_int = {}
        self.cached_amp = {}

    def build_cached_int(self, mcdata, mc_weight, batch=65000):
        mc_id = id(mcdata)
        if isinstance(mcdata, dict):
            mcdata = split_generator(mcdata, batch)
            mc_weight = split_generator(mc_weight, batch)
        dec = self.Amp.decay_group
        index, ret = None, []
        sum_weight = 1.0
        for data, weight in zip(mcdata, mc_weight):
            sum_weight += tf.reduce_sum(weight)
            index, a = opt_int.build_int_matrix(dec, data, weight)
            ret.append(a)

        int_matrix = tf.reduce_sum(ret, axis=0)

        @tf.function
        def int_mc():
            pm = opt_int.build_params_matrix(dec)
            ret = tf.reduce_sum(pm * int_matrix)
            return tf.math.real(ret)

        self.cached_int[mc_id] = int_mc

        # print(int_mc())
        # a = 0.0
        # for mc, w in zip(mcdata, mc_weight):
        #     a += tf.reduce_sum(self.Amp(mc) * w)
        # print(a)

    def get_cached_int(self, mc_id):
        return self.cached_int[mc_id]()

    # @tf.function
    def nll_grad_batch(self, data, mcdata, weight, mc_weight):
        """
        ``self.nll_grad()`` is replaced by this one???

        .. math::
          - \\frac{\\partial \\ln L}{\\partial \\theta_k } =
            -\\sum_{x_i \\in data } w_i \\frac{\\partial}{\\partial \\theta_k} \\ln f(x_i;\\theta_k)
            + (\\sum w_j ) \\left( \\frac{ \\partial }{\\partial \\theta_k} \\sum_{x_i \\in mc} f(x_i;\\theta_k) \\right)
              \\frac{1}{ \\sum_{x_i \\in mc} f(x_i;\\theta_k) }

        :param data:
        :param mcdata:
        :param weight:
        :param mc_weight:
        :return:
        """
        sw = tf.reduce_sum([tf.reduce_sum(i) for i in weight])
        data_id = id(data)
        data = list(data)
        weight = list(weight)
        if data_id not in self.cached_amp:
            self.cached_amp[data_id] = [
                build_amp.cached_amp2s(self.Amp.decay_group, i) for i in data
            ]
        ln_data, g_ln_data = sum_gradient(
            self.cached_amp[data_id],
            self.Amp.trainable_variables,
            weight=weight,
            trans=clip_log,
        )
        # print(ln_data, ln_data2, np.allclose(g_ln_data, g_ln_data2))
        mc_id = id(mcdata)
        if mc_id not in self.cached_int:
            self.build_cached_int(mcdata, mc_weight)
        with tf.GradientTape() as tape:
            int_mc = self.get_cached_int(mc_id)
        g_int_mc = tape.gradient(
            int_mc, self.Amp.trainable_variables, unconnected_gradients="zero"
        )

        # int_mc2, g_int_mc2 = sum_gradient(self.Amp, mcdata,
        #                                self.Amp.trainable_variables, weight=mc_weight)
        #
        # print("exp", int_mc, g_int_mc)
        # print("now", int_mc2, g_int_mc2)
        sw = tf.cast(sw, ln_data.dtype)

        g = list(
            map(lambda x: -x[0] + sw * x[1] / int_mc, zip(g_ln_data, g_int_mc))
        )
        nll = -ln_data + sw * tf.math.log(int_mc)

        return nll, g

    def nll_grad_hessian(
        self, data, mcdata, weight=1.0, batch=24000, bg=None, mc_weight=1.0
    ):
        """
        The parameters are the same with ``self.nll()``, but it will return Hessian as well.

        :return NLL: Real number. The value of NLL.
        :return gradients: List of real numbers. The gradients for each variable.
        :return Hessian: 2-D Array of real numbers. The Hessian matrix of the variables.
        """
        data, weight = self.get_weight_data(data, weight, bg=bg)
        if isinstance(mc_weight, float):
            mc_weight = tf.convert_to_tensor(
                [mc_weight] * data_shape(mcdata), dtype="float64"
            )
        n_mc = tf.reduce_sum(mc_weight)
        sw = tf.reduce_sum(weight)
        ln_data, g_ln_data, h_ln_data = sum_hessian(
            self.Amp,
            split_generator(data, batch),
            self.Amp.trainable_variables,
            weight=split_generator(weight, batch),
            trans=clip_log,
        )

        # int_mc, g_int_mc, h_int_mc = sum_hessian(self.Amp, split_generator(mcdata, batch),
        #                                         self.Amp.trainable_variables, weight=split_generator(
        #        mc_weight, batch))
        if isinstance(mc_weight, float):
            mc_weight = tf.convert_to_tensor(
                [mc_weight] * data_shape(mcdata), dtype="float64"
            )
            mc_weight = mc_weight / tf.reduce_sum(mc_weight)
        mc_id = id(mcdata)
        if mc_id not in self.cached_int:
            self.build_cached_int(mcdata, mc_weight)
        with tf.GradientTape(persistent=True) as tape0:
            with tf.GradientTape() as tape:
                y_i = self.get_cached_int(mc_id)
            g_i = tape.gradient(
                y_i, self.Amp.trainable_variables, unconnected_gradients="zero"
            )
        h_s_i = []
        for gi in g_i:
            # 2nd order derivative
            h_s_i.append(
                tape0.gradient(
                    gi,
                    self.Amp.trainable_variables,
                    unconnected_gradients="zero",
                )
            )
        del tape0
        int_mc = y_i
        g_int_mc = tf.convert_to_tensor(g_i)
        h_int_mc = tf.convert_to_tensor(h_s_i)

        n_var = len(g_ln_data)
        nll = -ln_data + sw * tf.math.log(int_mc / n_mc)
        g = -g_ln_data + sw * g_int_mc / int_mc

        g_int_mc = g_int_mc / int_mc
        g_outer = tf.reshape(g_int_mc, (-1, 1)) * tf.reshape(g_int_mc, (1, -1))

        h = -h_ln_data - sw * g_outer + sw / int_mc * h_int_mc
        return nll, g, h


class ModelCachedAmp(Model):
    """
    This class implements methods to calculate NLL as well as its derivatives for an amplitude model with Cached Int.
    It may include data for both signal and background.
    Cached Int well cause wrong results when float parameters include mass or width.

    :param amp: ``AllAmplitude`` object. The amplitude model.
    :param w_bkg: Real number. The weight of background.
    """

    def __init__(self, amp, w_bkg=1.0):
        self.Amp = amp
        self.w_bkg = w_bkg
        self.vm = amp.vm
        self.cached_amp = build_amp.build_amp2s(amp.decay_group)
        self.cached_data = {}

    # @tf.function
    def nll_grad_batch(self, data, mcdata, weight, mc_weight):
        """
        ``self.nll_grad()`` is replaced by this one???

        .. math::
          - \\frac{\\partial \\ln L}{\\partial \\theta_k } =
            -\\sum_{x_i \\in data } w_i \\frac{\\partial}{\\partial \\theta_k} \\ln f(x_i;\\theta_k)
            + (\\sum w_j ) \\left( \\frac{ \\partial }{\\partial \\theta_k} \\sum_{x_i \\in mc} f(x_i;\\theta_k) \\right)
              \\frac{1}{ \\sum_{x_i \\in mc} f(x_i;\\theta_k) }

        :param data:
        :param mcdata:
        :param weight:
        :param mc_weight:
        :return:
        """
        sw = tf.reduce_sum([tf.reduce_sum(i) for i in weight])
        data_id = id(data)
        data = list(data)
        weight = list(weight)
        if data_id not in self.cached_data:
            self.cached_data[data_id] = [
                build_amp.build_amp_matrix(self.Amp.decay_group, i)[1]
                for i in data
            ]
        ln_data, g_ln_data = sum_gradient_data2(
            self.cached_amp,
            self.Amp.trainable_variables,
            data,
            self.cached_data[data_id],
            weight=weight,
            trans=clip_log,
        )
        # print(ln_data, ln_data2, np.allclose(g_ln_data, g_ln_data2))
        mc_id = id(mcdata)
        mcdata = list(mcdata)
        if mc_id not in self.cached_data:
            self.cached_data[mc_id] = [
                build_amp.build_amp_matrix(self.Amp.decay_group, i)[1]
                for i in mcdata
            ]
        int_mc, g_int_mc = sum_gradient_data2(
            self.cached_amp,
            self.Amp.trainable_variables,
            mcdata,
            self.cached_data[mc_id],
            weight=mc_weight,
        )

        # int_mc2, g_int_mc2 = sum_gradient(self.Amp, mcdata,
        #                                self.Amp.trainable_variables, weight=mc_weight)
        #
        # print("exp", int_mc, g_int_mc)
        # print("now", int_mc2, g_int_mc2)
        sw = tf.cast(sw, ln_data.dtype)

        g = list(
            map(lambda x: -x[0] + sw * x[1] / int_mc, zip(g_ln_data, g_int_mc))
        )
        nll = -ln_data + sw * tf.math.log(int_mc)

        return nll, g
