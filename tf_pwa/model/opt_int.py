from .model import Model, sum_gradient, clip_log
from tf_pwa.experimental import opt_int
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
    
    def build_cached_int(self, mcdata, mc_weight):
        dec = self.Amp.decay_group
        index, ret = None, []
        sum_weight = 1.0
        for data, weight in zip(mcdata, mc_weight):
            sum_weight += tf.reduce_sum(weight)
            index, a = opt_int.build_int_matrix(dec, data, weight)
            ret.append(a)

        int_matrix = tf.reduce_sum(ret, axis=0)

        def int_mc():
            pm = opt_int.build_params_matrix(dec)
            ret = tf.reduce_sum(pm * int_matrix)
            return tf.math.real(ret)

        self.cached_int[id(mcdata)] = int_mc
        
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
