import numpy as np

from .model import Model, sum_gradient, clip_log, sum_hessian
from ..tensorflow_wrapper import tf
from ..data import data_shape, split_generator


class Model_cfit(Model):
    def __init__(self, amp, w_bkg=0.001, bg_f=None):
        self.Amp = amp
        if bg_f is None:
            bg_f = lambda x: tf.ones(shape=(data_shape(x),), dtype="float64")
        self.bg = bg_f
        self.vm = amp.vm
        self.w_bkg = w_bkg

    def nll(
        self,
        data,
        mcdata,
        weight: tf.Tensor = 1.0,
        batch=None,
        bg=None,
        mc_weight=None,
    ):
        """
        Calculate NLL.

        .. math::
          -\\ln L = -\\sum_{x_i \\in data } w_i \\ln P(x_i;\\theta_k) \\\\
          P(x_i;\\theta_k) = (1-f_{bg})Amp(x_i; \\theta_k) + f_{bg} Bg(x_{i};\\theta_k)

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
            int_mc = tf.math.log(tf.reduce_sum(mc_weight * self.Amp(mcdata)))
            int_bg = tf.math.log(tf.reduce_sum(mc_weight * self.bg(mcdata)))
        ln_data = tf.math.log(
            (1 - self.w_bkg) * sig_data / int_mc
            + self.w_bkg * bg_data / int_bg
        )
        nll_0 = -tf.reduce_sum(tf.cast(weight, ln_data.dtype) * ln_data)
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
        v_int_sig, v_int_bg = (
            tf.Variable(int_sig, dtype="float64"),
            tf.Variable(int_bg, dtype="float64"),
        )

        def prob(x):
            return (1 - self.w_bkg) * self.Amp(
                x
            ) / v_int_sig + self.w_bkg * self.bg(x) / v_int_bg

        ll, g_ll = sum_gradient(
            prob, data, var + [v_int_sig, v_int_bg], weight, trans=clip_log
        )
        g_ll_sig, g_ll_bg = g_ll[-2], g_ll[-1]
        g = [
            -g_ll[i] - g_int_sig[i] * g_ll_sig - g_int_bg[i] * g_ll_bg
            for i in range(len(var))
        ]
        return -ll, g

    def nll_grad_hessian(
        self, data, mcdata, weight=1.0, batch=24000, bg=None, mc_weight=1.0
    ):
        r"""
        The parameters are the same with ``self.nll()``, but it will return Hessian as well.

        .. math::
            \frac{\partial^2 f}{\partial x_i \partial x_j} =
            \frac{\partial y_k}{\partial x_i}
            \frac{\partial^2 f}{\partial y_k \partial y_l}
            \frac{\partial y_l}{\partial x_j}
            + \frac{\partial f}{\partial y_k} \frac{\partial^2 y_k}{\partial x_i \partial x_j}

        .. math::
            y = \{x_i; I_{sig}, I_{bg}\}

        .. math::
            \frac{\partial y_k }{\partial x_i} = (\delta_{ik};\frac{\partial I_{sig}}{\partial x_i}, \frac{\partial I_{bg}}{\partial x_i})

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
        var = self.vm.trainable_variables
        int_sig, g_int_sig, h_int_sig = sum_hessian(
            self.Amp,
            split_generator(mcdata, batch),
            var,
            weight=split_generator(mc_weight, batch),
        )

        int_bg, g_int_bg, h_int_bg = sum_hessian(
            self.bg,
            split_generator(mcdata, batch),
            var,
            weight=split_generator(mc_weight, batch),
        )

        v_int_sig, v_int_bg = (
            tf.Variable(int_sig, dtype="float64"),
            tf.Variable(int_bg, dtype="float64"),
        )

        def prob(x):
            return (1 - self.w_bkg) * self.Amp(
                x
            ) / v_int_sig + self.w_bkg * self.bg(x) / v_int_bg

        ll, g_ll, h_ll = sum_hessian(
            self.Amp,
            split_generator(data, batch),
            var + [v_int_sig, v_int_bg],
            weight=split_generator(weight, batch),
            trans=clip_log,
        )

        n_var = len(var)
        g_ll_sig, g_ll_bg = g_ll[-2], g_ll[-1]
        g = [
            -g_ll[i] - g_int_sig[i] * g_ll_sig - g_int_bg[i] * g_ll_bg
            for i in range(len(var))
        ]
        jac = np.concatenate([np.eye(n_var), [g_int_sig, g_int_bg]], axis=0)
        h = (
            np.dot(jac.T, np.dot(h_ll, jac))
            + g_ll_sig * h_int_sig
            + g_ll_bg * h_int_bg
        )
        return -ll, g, -h
