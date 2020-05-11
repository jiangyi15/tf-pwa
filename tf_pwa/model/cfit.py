from .model import Model, sum_gradient, clip_log
from ..tensorflow_wrapper import tf
from ..data import data_shape


class Model_cfit(Model):
    def __init__(self, amp, w_bkg=0.1, bg_f=None):
        self.Amp = amp
        if bg_f is None:
            bg_f = lambda x: tf.ones(shape=(data_shape(x),), dtype="float64")
        self.bg = bg_f
        self.vm = amp.vm
        self.w_bkg = w_bkg

    def nll(self, data, mcdata, weight: tf.Tensor = 1.0, batch=None, bg=None, mc_weight=None):
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
        sig_data = self.Amp(data)
        bg_data = self.bg(data)
        if mc_weight is None:
            int_mc = tf.reduce_mean(self.Amp(mcdata))
            int_bg = tf.reduce_mean(self.bg(mcdata))
        else:
            int_mc = tf.math.log(tf.reduce_sum(mc_weight*self.Amp(mcdata)))
            int_bg = tf.math.log(tf.reduce_sum(mc_weight*self.bg(mcdata)))
        ln_data = tf.math.log((1-self.w_bkg) * sig_data / int_mc + self.w_bkg * bg_data /int_bg)
        nll_0 = - tf.reduce_sum(tf.cast(weight, ln_data.dtype) * ln_data)
        return nll_0

    def nll_grad_batch(self, data, mcdata, weight, mc_weight):
        r"""
        .. math::
            P = (1-frac) \frac{amp(data)}{\sum amp(phsp)} + frac \frac{bg(data)}{\sum bg(phsp)}
        
        .. math::
            nll = - \sum log(p)
        
        .. math::
           \frac{\partial nll}{\partial \theta} = -\sum \frac{1}{p} \frac{\partial p}{\partial \theta}
           = - \sum \frac{\partial \ln \bar{p}}{\partial \theta }  +\frac{\partial nll}{\partial I_{sig} }
           \frac{\partial I_{sig} }{\partial \theta} 
           +\frac{\partial nll}{\partial I_{bg}}\frac{\partial I_{sig}}{\partial \theta}

        """
        var = self.vm.trainable_variables
        mcdata = list(mcdata)
        mc_weight = list(mc_weight)
        int_sig, g_int_sig = sum_gradient(self.Amp, mcdata, var, mc_weight)
        int_bg, g_int_bg = sum_gradient(self.bg, mcdata, var, mc_weight)
        v_int_sig, v_int_bg = tf.Variable(int_sig, dtype="float64"), tf.Variable(int_bg, dtype="float64")
        def prob(x):
            return (1-self.w_bkg) * self.Amp(x)/v_int_sig + self.w_bkg * self.bg(x)/v_int_bg
        ll, g_ll = sum_gradient(prob, data, var + [v_int_sig, v_int_bg], weight, trans=clip_log)
        g_ll_sig, g_ll_bg = g_ll[-2], g_ll[-1]
        g = [-g_ll[i] - g_int_sig[i] * g_ll_sig - g_int_bg[i] * g_ll_bg for i in range(len(var))]
        return - ll, g


