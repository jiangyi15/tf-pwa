from .model import Model, sum_gradient, clip_log, sum_hessian, split_generator
from tf_pwa.experimental import opt_int
from tf_pwa.data import data_shape
import tensorflow as tf


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
        ln_data, g_ln_data = sum_gradient(self.Amp, data,
                                          self.Amp.trainable_variables, weight=weight, trans=clip_log)
        mc_id = id(mcdata)
        if mc_id not in self.cached_int:
            self.build_cached_int(mcdata, mc_weight)
        with tf.GradientTape() as tape:
            int_mc = self.get_cached_int(mc_id)
        g_int_mc = tape.gradient(int_mc, self.Amp.trainable_variables, unconnected_gradients="zero")

        # int_mc2, g_int_mc2 = sum_gradient(self.Amp, mcdata,
        #                                self.Amp.trainable_variables, weight=mc_weight)
        #
        # print("exp", int_mc, g_int_mc)
        # print("now", int_mc2, g_int_mc2)
        sw = tf.cast(sw, ln_data.dtype)

        g = list(map(lambda x: - x[0] + sw * x[1] /
                               int_mc, zip(g_ln_data, g_int_mc)))
        nll = - ln_data + sw * tf.math.log(int_mc)
        return nll, g

    def nll_grad_hessian(self, data, mcdata, weight=1.0, batch=24000, bg=None, mc_weight=1.0):
        """
        The parameters are the same with ``self.nll()``, but it will return Hessian as well.

        :return NLL: Real number. The value of NLL.
        :return gradients: List of real numbers. The gradients for each variable.
        :return Hessian: 2-D Array of real numbers. The Hessian matrix of the variables.
        """
        data, weight = self.get_weight_data(data, weight, bg=bg)
        if isinstance(mc_weight, float):
            mc_weight = tf.convert_to_tensor([mc_weight] * data_shape(mcdata), dtype="float64")
        n_mc = tf.reduce_sum(mc_weight)
        sw = tf.reduce_sum(weight)
        ln_data, g_ln_data, h_ln_data = sum_hessian(self.Amp, split_generator(data, batch),
                                                    self.Amp.trainable_variables,
                                                    weight=split_generator(weight, batch), trans=clip_log)

        #int_mc, g_int_mc, h_int_mc = sum_hessian(self.Amp, split_generator(mcdata, batch),
        #                                         self.Amp.trainable_variables, weight=split_generator(
        #        mc_weight, batch))
        if isinstance(mc_weight, float):
            mc_weight = tf.convert_to_tensor([mc_weight] * data_shape(mcdata), dtype="float64")
            mc_weight = mc_weight / tf.reduce_sum(mc_weight)
        mc_id = id(mcdata)
        if mc_id not in self.cached_int:
            self.build_cached_int(mcdata, mc_weight)
        with tf.GradientTape(persistent=True) as tape0:
            with tf.GradientTape() as tape:
                y_i = self.get_cached_int(mc_id)
            g_i = tape.gradient(y_i, self.Amp.trainable_variables, unconnected_gradients="zero")
        h_s_i = []
        for gi in g_i:
            # 2nd order derivative
            h_s_i.append(tape0.gradient(gi, self.Amp.trainable_variables, unconnected_gradients="zero"))
        del tape0
        int_mc = y_i
        g_int_mc = tf.convert_to_tensor(g_i)
        h_int_mc = tf.convert_to_tensor(h_s_i)

        n_var = len(g_ln_data)
        nll = - ln_data + sw * tf.math.log(int_mc/n_mc)
        g = - g_ln_data + sw * g_int_mc / int_mc

        g_int_mc = g_int_mc / int_mc
        g_outer = tf.reshape(g_int_mc, (-1, 1)) * tf.reshape(g_int_mc, (1, -1))

        h = - h_ln_data - sw * g_outer + sw / int_mc * h_int_mc
        return nll, g, h
