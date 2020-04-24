"""
This module provides methods to calculate NLL(Negative Log-Likelihood) as well as its derivatives.
"""

import numpy as np
import math
from .data import data_shape, split_generator, data_merge, data_split
from .tensorflow_wrapper import tf
from .utils import time_print
from .config import get_config
from .variable import Variable


def loop_generator(var):
    """
    Put to utils.py???

    :param var:
    :return:
    """
    while True:
        yield var


def sum_gradient(f, data, var, weight=1.0, trans=tf.identity, args=(), kwargs=None):
    """
    NLL is the sum of trans(f(data)):math:`*`weight; gradient is the derivatives for each variable in ``var``.

    :param f: Function. The amplitude PDF.
    :param data: Data array
    :param var: List of strings. Names of the trainable variables in the PDF.
    :param weight: Weight factor for each data point. It's either a real number or an array of the same shape with ``data``.
    :param trans: Function. Transformation of ``data`` before multiplied by ``weight``.
    :param kwargs: Further arguments for ``f``.
    :return: Real number NLL, list gradient
    """
    kwargs = kwargs if kwargs is not None else {}
    if isinstance(weight, float):
        weight = loop_generator(weight)
    ys = []
    gs = []
    for data_i, weight_i in zip(data, weight):
        with tf.GradientTape() as tape:
            part_y = trans(f(data_i, *args, **kwargs))
            y_i = tf.reduce_sum(tf.cast(weight_i, part_y.dtype) * part_y)
        g_i = tape.gradient(y_i, var, unconnected_gradients="zero")
        ys.append(y_i)
        gs.append(g_i)
    nll = sum(ys)
    g = list(map(sum, zip(*gs)))
    return nll, g

def sum_hessian(f, data, var, weight=1.0, trans=tf.identity, args=(), kwargs=None):
    """
    The parameters are the same with ``sum_gradient()``, but this function will return hessian as well,
    which is the matrix of the second-order derivative.

    :return: Real number NLL, list gradient, 2-D list hessian
    """
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
            g_i = tape.gradient(y_i, var, unconnected_gradients="zero")
        h_s_i = []
        for gi in g_i:
            # 2nd order derivative
            h_s_i.append(tape0.gradient(gi, var, unconnected_gradients="zero"))
        del tape0
        y_s.append(y_i)
        g_s.append(g_i)
        h_s.append(h_s_i)
    nll = tf.reduce_sum(y_s)
    g = tf.reduce_sum(g_s, axis=0)
    h = tf.reduce_sum(h_s, axis=0)
    # h = [[sum(j) for j in zip(*i)] for i in h_s]
    return nll, g, h

def sum_gradient_new(amp, data, mcdata, weight, mcweight, var, trans=tf.math.log, w_flatmc=lambda:0, args=(), kwargs=None):
    """
    NLL is the sum of trans(f(data)):math:`*`weight; gradient is the derivatives for each variable in ``var``.

    :param f: Function. The amplitude PDF.
    :param data: Data array
    :param var: List of strings. Names of the trainable variables in the PDF.
    :param weight: Weight factor for each data point. It's either a real number or an array of the same shape with ``data``.
    :param trans: Function. Transformation of ``data`` before multiplied by ``weight``.
    :param kwargs: Further arguments for ``f``.
    :return: Real number NLL, list gradient
    """
    kwargs = kwargs if kwargs is not None else {}
    if isinstance(weight, float):
        weight = loop_generator(weight)
    ys = []
    #gs = []
    ymc = []
    with tf.GradientTape() as tape:
        for mcdata_i, mcweight_i in zip(mcdata, mcweight):
            part_y = amp(mcdata_i, *args, **kwargs)
            y_i = tf.reduce_sum(tf.cast(mcweight_i, part_y.dtype) * part_y)
            ymc.append(y_i)
        int_dt = tf.reduce_sum(ymc)
        for data_i, weight_i in zip(data, weight):
            wmc = w_flatmc()
            part_y = (amp(data_i, *args, **kwargs)/int_dt + wmc) / (1+wmc)
            part_y = trans(part_y)
            y_i = tf.reduce_sum(tf.cast(weight_i, part_y.dtype) * part_y)
            ys.append(y_i)
        nll = -tf.reduce_sum(ys)
    g = tape.gradient(nll, var, unconnected_gradients="zero")
    return nll, g

def sum_hessian_new(amp, data, mcdata, weight, mcweight, var, trans=tf.math.log, w_flatmc=lambda:0, args=(), kwargs=None):
    """
    The parameters are the same with ``sum_gradient()``, but this function will return hessian as well,
    which is the matrix of the second-order derivative.

    :return: Real number NLL, list gradient, 2-D list hessian
    """
    kwargs = kwargs if kwargs is not None else {}
    if isinstance(weight, float):
        weight = loop_generator(weight)
    ys = []
    ymc = []
    with tf.GradientTape(persistent=True) as tape0:
        with tf.GradientTape() as tape:
            for mcdata_i, mcweight_i in zip(mcdata, mcweight):
                part_y = amp(mcdata_i, *args, **kwargs)
                y_i = tf.reduce_sum(tf.cast(mcweight_i, part_y.dtype) * part_y)
                ymc.append(y_i)
            int_dt = tf.reduce_sum(ymc)
            for data_i, weight_i in zip(data, weight):
                wmc = w_flatmc()
                part_y = (amp(data_i, *args, **kwargs)/int_dt + wmc) / tf.math.sqrt(1+wmc*wmc)
                part_y = trans(part_y)
                y_i = tf.reduce_sum(tf.cast(weight_i, part_y.dtype) * part_y)
                ys.append(y_i)
            nll = -tf.reduce_sum(ys)
        gradient = tape.gradient(nll, var, unconnected_gradients="zero")
    hessian = []
    for gi in gradient:
        # 2nd order derivative
        hessian.append(tape0.gradient(gi, var, unconnected_gradients="zero"))
    del tape0
    return nll, gradient, hessian

def clip_log(x, _epsilon=1e-6):
    """clip log to allowed large value"""
    x_cut = tf.where(x > _epsilon, x, tf.ones_like(x) * _epsilon)
    b_t = tf.math.log(x)
    delta_x = x - _epsilon
    b_f = np.log(_epsilon) + delta_x / _epsilon - (delta_x / _epsilon) ** 2 / 2.0
    print("$$$", tf.where(x < _epsilon).numpy().tolist())
    return tf.where(x > _epsilon, b_t, b_f)

class Model(object):
    """
    This class implements methods to calculate NLL as well as its derivatives for an amplitude model. It may include
    data for both signal and background.

    :param amp: ``AllAmplitude`` object. The amplitude model.
    :param w_bkg: Real number. The weight of background.
    """

    def __init__(self, amp, w_bkg=1.0):
        self.Amp = amp
        self.w_bkg = w_bkg

    def get_weight_data(self, data, weight=None, bg=None, alpha=True):
        """
        Blend data and background data together multiplied by their weights.

        :param data: Data array
        :param weight: Weight for data
        :param bg: Data array for background
        :param alpha: Boolean. If it's true, ``weight`` will be multiplied by a factor :math:`\\alpha=`???
        :return: Data, weight. Their length both equals ``len(data)+len(bg)``.
        """
        if weight is None:
            weight = data.get("weight", 1.0)
        if isinstance(weight, float):
            n_data = data_shape(data)
            weight = tf.convert_to_tensor(
                [weight] * n_data, dtype=get_config("dtype"))
        if bg is not None:
            n_bg = data_shape(bg)
            data = data_merge(data, bg)
            bg_weight = bg.get("weight", None)
            if bg_weight is None:
                bg_weight = tf.convert_to_tensor(
                    [-self.w_bkg] * n_bg, dtype=get_config("dtype"))
            else:
                bg_weight = tf.convert_to_tensor(bg_weight, dtype=get_config("dtype"))
            weight = tf.concat([weight, bg_weight], axis=0)
        if alpha:
            alpha = tf.reduce_sum(weight) / tf.reduce_sum(weight * weight)
            return data, alpha * weight
        return data, weight

    def nll(self, data, mcdata, weight: tf.Tensor = 1.0, batch=None, bg=None):
        """
        Calculate NLL.

        .. math::
          -\\ln L = -\\sum_{x_i \\in data } w_i \\ln f(x_i;\\theta_k) +  (\\sum w_j ) \\ln \\sum_{x_i \\in mc } f(x_i;\\theta_k)

        :param data: Data array
        :param mcdata: MCdata array
        :param weight: Weight of data???
        :param batch: The length of array to calculate as a vector at a time. How to fold the data array may depend on the GPU computability.
        :param bg: Background data array. It can be set to ``None`` if there is no such thing.
        :return: Real number. The value of NLL.
        """
        data, weight = self.get_weight_data(data, weight, bg=bg)
        sw = tf.reduce_sum(weight)
        ln_data = tf.math.log(self.Amp(data))
        int_mc = tf.math.log(tf.reduce_mean(self.Amp(mcdata)))
        nll_0 = - tf.reduce_sum(tf.cast(weight, ln_data.dtype) * ln_data)
        return nll_0 + tf.cast(sw, int_mc.dtype) * int_mc

    def nll_grad(self, data, mcdata, weight=1.0, batch=65000, bg=None):
        """
        Calculate NLL and its gradients.

        .. math::
          - \\frac{\\partial \\ln L}{\\partial \\theta_k } =
            -\\sum_{x_i \\in data } w_i \\frac{\\partial}{\\partial \\theta_k} \\ln f(x_i;\\theta_k)
            + (\\sum w_j ) \\left( \\frac{ \\partial }{\\partial \\theta_k} \\sum_{x_i \\in mc} f(x_i;\\theta_k) \\right)
              \\frac{1}{ \\sum_{x_i \\in mc} f(x_i;\\theta_k) }

        The parameters are the same with ``self.nll()``, but it will return gradients as well.

        :return NLL: Real number. The value of NLL.
        :return gradients: List of real numbers. The gradients for each variable.
        """
        data, weight = self.get_weight_data(data, weight, bg=bg)
        n_mc = data_shape(mcdata)
        sw = tf.reduce_sum(weight)
        ln_data, g_ln_data = sum_gradient(self.Amp, split_generator(data, batch),
                                          self.Amp.trainable_variables, weight=split_generator(
                weight, batch),
                                          trans=clip_log)
        int_mc, g_int_mc = sum_gradient(self.Amp, split_generator(mcdata, batch),
                                        self.Amp.trainable_variables, weight=1 / n_mc)

        sw = tf.cast(sw, ln_data.dtype)

        g = list(map(lambda x: - x[0] + sw * x[1] /
                               int_mc, zip(g_ln_data, g_int_mc)))
        nll = - ln_data + sw * tf.math.log(int_mc)
        return nll, g

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
        ln_data, g_ln_data = sum_gradient(self.Amp, data,
                                          self.Amp.trainable_variables, weight=weight, trans=clip_log)
        int_mc, g_int_mc = sum_gradient(self.Amp, mcdata,
                                        self.Amp.trainable_variables, weight=mc_weight)

        sw = tf.cast(sw, ln_data.dtype)

        g = list(map(lambda x: - x[0] + sw * x[1] /
                               int_mc, zip(g_ln_data, g_int_mc)))
        nll = - ln_data + sw * tf.math.log(int_mc)
        return nll, g

    def nll_grad_hessian(self, data, mcdata, weight=1.0, batch=24000, bg=None):
        """
        The parameters are the same with ``self.nll()``, but it will return Hessian as well.

        :return NLL: Real number. The value of NLL.
        :return gradients: List of real numbers. The gradients for each variable.
        :return Hessian: 2-D Array of real numbers. The Hessian matrix of the variables.
        """
        data, weight = self.get_weight_data(data, weight, bg=bg)
        n_mc = data_shape(mcdata)
        sw = tf.reduce_sum(weight)
        ln_data, g_ln_data, h_ln_data = sum_hessian(self.Amp, split_generator(data, batch),
                                                    self.Amp.trainable_variables, weight=split_generator(
                weight, batch),
                                                    trans=tf.math.log)
        int_mc, g_int_mc, h_int_mc = sum_hessian(self.Amp, split_generator(mcdata, batch),
                                                 self.Amp.trainable_variables)

        n_var = len(g_ln_data)
        nll = - ln_data + sw * tf.math.log(int_mc / n_mc)
        g = - g_ln_data + sw * g_int_mc / int_mc

        g_int_mc = g_int_mc / int_mc
        g_outer = tf.reshape(g_int_mc, (-1, 1)) * tf.reshape(g_int_mc, (1, -1))

        h = - h_ln_data - sw * g_outer + sw / int_mc * h_int_mc
        return nll, g, h

    def set_params(self, var):
        """
        It has interface to ``Amplitude.set_params()``.
        """
        self.Amp.set_params(var)

    def get_params(self, trainable_only=False):
        """
        It has interface to ``Amplitude.get_params()``.
        """
        return self.Amp.get_params(trainable_only)


class Model_new(Model):
    """
    This class implements methods to calculate NLL as well as its derivatives for an amplitude model. It may include
    data for both signal and background.

    :param amp: ``AllAmplitude`` object. The amplitude model.
    :param w_bkg: Real number. The weight of background.
    """
    def __init__(self, amp, w_bkg=1.0, w_inmc=0, float_wmc=False):
        super(Model_new, self).__init__(amp, w_bkg)
        #self.w_inmc = w_inmc
        self.w_inmc = Variable("weight_injectMC", value=w_inmc, vm=self.Amp.vm)
        if not float_wmc:
            self.w_inmc.fixed()

    def get_weight_data(self, data, weight=1.0, bg=None, inmc=None, alpha=True):
        """
        Blend data and background data together multiplied by their weights.

        :param data: Data array
        :param weight: Weight for data
        :param bg: Data array for background
        :param alpha: Boolean. If it's true, ``weight`` will be multiplied by a factor :math:`\\alpha=`???
        :return: Data, weight. Their length both equals ``len(data)+len(bg)``.
        """
        n_data = data_shape(data)
        if isinstance(weight, float):
            weight = tf.convert_to_tensor(
                [weight] * n_data, dtype=get_config("dtype"))
        if bg is not None:
            n_bg = data_shape(bg)
            data = data_merge(data, bg)
            bg_weight = tf.convert_to_tensor(
                [-self.w_bkg] * n_bg, dtype=get_config("dtype"))
            weight = tf.concat([weight, bg_weight], axis=0)
        if inmc is not None:
            n_inmc = data_shape(inmc)
            data = data_merge(data, inmc)
            wmc = self.w_inmc() * n_data / n_inmc
            inmc_weight = tf.convert_to_tensor(
                [wmc] * n_inmc, dtype=get_config("dtype"))
            weight = tf.concat([weight, inmc_weight], axis=0)

        if alpha:
            alpha = tf.reduce_sum(weight) / tf.reduce_sum(weight * weight) # correct with inject MC?
            return data, alpha * weight
        return data, weight

    def nll(self, data, mcdata, weight: tf.Tensor = 1.0, batch=None, bg=None):
        """
        Calculate NLL.

        .. math::
          -\\ln L = -\\sum_{x_i \\in data } w_i \\ln f(x_i;\\theta_k) +  (\\sum w_j ) \\ln \\sum_{x_i \\in mc } f(x_i;\\theta_k)

        :param data: Data array
        :param mcdata: MCdata array
        :param weight: Weight of data???
        :param batch: The length of array to calculate as a vector at a time. How to fold the data array may depend on the GPU computability.
        :param bg: Background data array. It can be set to ``None`` if there is no such thing.
        :return: Real number. The value of NLL.
        """
        data, weight = self.get_weight_data(data, weight, bg=bg)
        sw = tf.reduce_sum(weight)
        ln_data = tf.math.log(self.Amp(data))
        int_mc = tf.math.log(tf.reduce_mean(self.Amp(mcdata)))
        nll_0 = - tf.reduce_sum(tf.cast(weight, ln_data.dtype) * ln_data)
        return nll_0 + tf.cast(sw, int_mc.dtype) * int_mc

    def nll_grad_batch(self, data, mcdata, weight, mc_weight):
        """
        ``self.nll_grad_new``
        """
        #N_data = 2525
        #N_flatmc = 3106
        #w_flatmc = 0#878/2525#2889/2525#3106/2525
        nll, g = sum_gradient_new(self.Amp, data, mcdata, weight, mc_weight,
                                          self.Amp.trainable_variables, clip_log, self.w_inmc)
        #print("@@@",nll,np.array(g).tolist())
        return nll, g

    def nll_grad_hessian(self, data, mcdata, weight, mc_weight):
        """
        The parameters are the same with ``self.nll()``, but it will return Hessian as well.

        :return NLL: Real number. The value of NLL.
        :return gradients: List of real numbers. The gradients for each variable.
        :return Hessian: 2-D Array of real numbers. The Hessian matrix of the variables.
        """
        nll, g, h = sum_hessian_new(self.Amp, data, mcdata, weight, mc_weight,
                                          self.Amp.trainable_variables, clip_log, self.w_inmc)
        h = tf.stack(h)
        return nll, g, h


class ConstrainModel(Model):
    """
    negative log likelihood model with constrains

    """

    def __init__(self, amp, w_bkg=1.0, constrain={}):
        super(ConstrainModel, self).__init__(amp, w_bkg)
        self.constrain = constrain  # priori gauss constrain for the fitting parameters

    def get_constrain_term(self):  # the priori constrain term added to NLL
        r"""
        constrain: Gauss(mean,sigma) 
          by add a term :math:`\frac{(\theta_i-\bar{\theta_i})^2}{2\sigma^2}`

        """
        t_var = self.Amp.trainable_variables
        t_var_name = [i.name for i in t_var]
        var_dict = dict(zip(t_var_name, t_var))
        nll = 0.0
        for i in self.constrain:
            if not i in var_dict:
                break
            pi = self.constrain[i]
            if isinstance(pi, tuple) and len(pi) == 2:
                mean, sigma = pi
                var = var_dict[i]
                nll += (var - mean) ** 2 / (sigma ** 2) / 2
        return nll

    def get_constrain_grad(self):  # the constrained parameter's 1st differentiation
        r"""
        constrain: Gauss(mean,sigma) 
          by add a term :math:`\frac{d}{d\theta_i}\frac{(\theta_i-\bar{\theta_i})^2}{2\sigma^2} = \frac{\theta_i-\bar{\theta_i}}{\sigma^2}`

        """
        t_var = self.Amp.trainable_variables
        t_var_name = [i.name for i in t_var]
        var_dict = dict(zip(t_var_name, t_var))
        g_dict = {}
        for i in self.constrain:
            if not i in var_dict:
                break
            pi = self.constrain[i]
            if isinstance(pi, tuple) and len(pi) == 2:
                mean, sigma = pi
                var = var_dict[i]
                g_dict[i] = (var - mean) / (sigma ** 2)  # 1st differentiation
        nll_g = []
        for i in t_var_name:
            if i in g_dict:
                nll_g.append(g_dict[i])
            else:
                nll_g.append(0.0)
        return nll_g

    def get_constrain_hessian(self):
        """the constrained parameter's 2nd differentiation"""
        t_var = self.Amp.trainable_variables
        t_var_name = [i.name for i in t_var]
        var_dict = dict(zip(t_var_name, t_var))
        g_dict = {}
        for i in self.constrain:
            if not i in var_dict:
                break
            pi = self.constrain[i]
            if isinstance(pi, tuple) and len(pi) == 2:
                mean, sigma = pi
                var = var_dict[i]
                g_dict[i] = 1 / (sigma ** 2)  # 2nd differentiation
        nll_g = []
        for i in t_var_name:
            if i in g_dict:
                nll_g.append(g_dict[i])
            else:
                nll_g.append(0.0)
        return np.diag(nll_g)

    def nll(self, data, mcdata, weight=1.0, bg=None, batch=None):
        r"""
        calculate negative log-likelihood 

        .. math::
          -\ln L = -\sum_{x_i \in data } w_i \ln f(x_i;\theta_i) +  (\sum w_i ) \ln \sum_{x_i \in mc } f(x_i;\theta_i) + cons

        """
        nll_0 = super(ConstrainModel, self).nll(
            data, mcdata, weight=weight, batch=batch, bg=bg)
        cons = self.get_constrain_term()
        return nll_0 + cons

    def nll_gradient(self, data, mcdata, weight=1.0, batch=None, bg=None):
        r"""
        calculate negative log-likelihood with gradient

        .. math::
          \frac{\partial }{\partial \theta_i }(-\ln L) = -\sum_{x_i \in data } w_i \frac{\partial }{\partial \theta_i } \ln f(x_i;\theta_i) +  
          \frac{\sum w_i }{\sum_{x_i \in mc }f(x_i;\theta_i)} \sum_{x_i \in mc } \frac{\partial }{\partial \theta_i } f(x_i;\theta_i) + cons

        """
        cons_grad = self.get_constrain_grad()  # the constrain term
        cons = self.get_constrain_term()
        nll0, g0 = super(ConstrainModel, self).nll_grad(data, mcdata, weight=weight, batch=batch, bg=bg)
        nll = nll0 + cons
        g = [cons_grad[i] + g0[i] for i in range(len(g0))]
        return nll, g


class FCN(object):
    """
    This class implements methods to calculate the NLL as well as its derivatives for a general function.

    :param model: Model object.
    :param data: Data array.
    :param mcdata: MCdata array.
    :param bg: Background array.
    :param batch: The length of array to calculate as a vector at a time. How to fold the data array may depend on the GPU computability.
    """

    def __init__(self, model, data, mcdata, bg=None, batch=65000, inmc=None):
        self.model = model
        self.n_call = 0
        self.n_grad = 0
        self.cached_nll = None
        if inmc is None:
            data, weight = self.model.get_weight_data(data, bg=bg)
            print("Using Model_old")
        else:
            data, weight = self.model.get_weight_data(data, bg=bg, inmc=inmc)
            print("Using Model_new")
        n_mcdata = data_shape(mcdata)
        self.alpha = tf.reduce_sum(weight) / tf.reduce_sum(weight * weight)
        self.weight = weight
        self.data = data
        self.batch_data = list(split_generator(data, batch))
        self.mcdata = mcdata
        self.batch_mcdata = list(split_generator(mcdata, batch))
        self.batch = batch
        if "weight" in mcdata:
            self.mc_weight = tf.convert_to_tensor(mcdata["weight"], dtype="float64")
        else:
            self.mc_weight = tf.convert_to_tensor(
                [1 / n_mcdata] * n_mcdata, dtype="float64")

    # @time_print
    def __call__(self, x):
        """
        :param x: List. Values of variables.
        :return nll: Real number. The value of NLL.
        """
        self.model.set_params(x)
        if type(self.model) == Model_new:
            nll, g = self.nll_grad(x)
        else:
            nll = self.model.nll(self.data, self.mcdata, weight=self.weight)
            self.cached_nll = nll
            self.n_call += 1
        return nll

    def grad(self, x):
        """
        :param x: List. Values of variables.
        :return gradients: List of real numbers. The gradients for each variable.
        """
        nll, g = self.nll_grad(x)
        return g

    @time_print
    def nll_grad(self, x):
        """
        :param x: List. Values of variables.
        :return nll: Real number. The value of NLL.
        :return gradients: List of real numbers. The gradients for each variable.
        """
        self.model.set_params(x)
        nll, g = self.model.nll_grad_batch(self.batch_data, self.batch_mcdata,
                                           weight=list(data_split(self.weight, self.batch)),
                                           mc_weight=data_split(self.mc_weight, self.batch))
        self.cached_nll = nll
        self.n_call += 1
        return nll, np.array(g)

    def nll_grad_hessian(self, x, batch=None):
        """
        :param x: List. Values of variables.
        :return nll: Real number. The value of NLL.
        :return gradients: List of real numbers. The gradients for each variable.
        :return hessian: 2-D Array of real numbers. The Hessian matrix of the variables.
        """
        self.model.set_params(x)
        if type(self.model) == Model_new:
            nll, g, h = self.model.nll_grad_hessian(self.batch_data, self.batch_mcdata,
                                           weight=list(data_split(self.weight, self.batch)),
                                           mc_weight=data_split(self.mc_weight, self.batch))
        else:
            if batch is None:
                batch = self.batch
            nll, g, h = self.model.nll_grad_hessian(self.data, self.mcdata,
                                                    weight=self.weight, batch=batch)

        return nll, g, h


class CombineFCN(object):
    """
    This class implements methods to calculate the NLL as well as its derivatives for a general function.

    :param model: List of model object.
    :param data: List of data array.
    :param mcdata: list of MCdata array.
    :param bg: list of Background array.
    :param batch: The length of array to calculate as a vector at a time. How to fold the data array may depend on the GPU computability.
    """

    def __init__(self, model=None, data=None, mcdata=None, bg=None, fcns=None, batch=65000):
        if fcns is None:
            assert model is not None, "model required"
            assert data is not None, "data required"
            assert mcdata is not None, "mcdata required"
            self.fcns = []
            self.cached_nll = 0.0
            if bg is None:
                bg = loop_generator(None)
            for model_i, data_i, mcdata_i, bg_i in zip(model, data, mcdata, bg):
                self.fcns.append(FCN(model_i, data_i, mcdata_i, bg_i))
        else:
            self.fcns = list(fcns)

    # @time_print
    def __call__(self, x):
        """
        :param x: List. Values of variables.
        :return nll: Real number. The value of NLL.
        """
        nlls = []
        for i in self.fcns:
            nlls.append(i(x))
        self.cached_nll = sum(nlls)
        return self.cached_nll

    def grad(self, x):
        """
        :param x: List. Values of variables.
        :return gradients: List of real numbers. The gradients for each variable.
        """
        gs = []
        for i in self.fcns:
            g = i.grad(x)
            gs.append(g)
        return sum(gs)

    @time_print
    def nll_grad(self, x):
        """
        :param x: List. Values of variables.
        :return nll: Real number. The value of NLL.
        :return gradients: List of real numbers. The gradients for each variable.
        """
        nlls = []
        gs = []
        for i in self.fcns:
            nll, g = i.nll_grad(x)
            nlls.append(nll)
            gs.append(g)
        self.cached_nll = sum(nlls)
        return self.cached_nll, sum(gs)

    def nll_grad_hessian(self, x, batch=None):
        """
        :param x: List. Values of variables.
        :return nll: Real number. The value of NLL.
        :return gradients: List of real numbers. The gradients for each variable.
        :return hessian: 2-D Array of real numbers. The Hessian matrix of the variables.
        """
        nlls = []
        gs = []
        hs = []
        for i in self.fcns:
            nll, g, h = i.nll_grad_hessian(x)
            nlls.append(nll)
            gs.append(g)
            hs.append(h)
        return sum(nlls), sum(gs), sum(hs)
