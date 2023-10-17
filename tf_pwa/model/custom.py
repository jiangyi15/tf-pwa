import numpy as np
import tensorflow as tf

from tf_pwa.model import Model, register_nll_model
from tf_pwa.variable import SumVar

"""
Custom nll model
"""


class BaseCustomModel(Model):
    def nll(
        self,
        data,
        mcdata,
        weight: tf.Tensor = 1.0,
        batch=None,
        bg=None,
        mc_weight=1.0,
    ):
        return self.nll_grad_batch(
            [data], [mcdata], [weight], mc_weight=[mc_weight]
        )[0]

    def eval_normal_factors(self, mcdata, weight=None):
        return []

    def eval_nll_part(self, data, weight=None, norm=None):
        raise NotImplementedError("")

    def nll_grad_batch(self, data, mcdata, weight, mc_weight):
        all_var = self.Amp.trainable_variables
        n_var = len(all_var)
        int_mc = None  # SumVar(0., np.zeros((n_var,)), all_var)
        for i, j in zip(mcdata, mc_weight):
            with tf.GradientTape(persistent=True) as tape:
                a = self.eval_normal_factors(i, j)
            grads = [
                tape.gradient(ai, all_var, unconnected_gradients="zero")
                for ai in a
            ]
            del tape
            tmp = [
                SumVar(ai, tf.stack(gi), all_var) for ai, gi in zip(a, grads)
            ]
            if int_mc is None:
                int_mc = tmp
            else:
                int_mc = [a + b for a, b in zip(int_mc, tmp)]
        ret = 0.0
        ret_grad = 0.0
        for i, j in zip(data, weight):
            with tf.GradientTape() as tape:
                if int_mc is None:
                    a = self.eval_nll_part(i, j, None)
                else:
                    a = self.eval_nll_part(i, j, [k() for k in int_mc])
            grads = tape.gradient(a, all_var, unconnected_gradients="zero")
            ret = ret + a
            ret_grad = ret_grad + tf.stack(grads)
        return ret, ret_grad

    def nll_grad_hessian(
        self, data, mcdata, weight=1.0, batch=24000, bg=None, mc_weight=1.0
    ):
        pass


@register_nll_model("simple")
class SimpleNllModel(BaseCustomModel):
    def eval_normal_factors(self, mcdata, weight):
        return [tf.reduce_sum(self.Amp(mcdata) * weight)]

    def eval_nll_part(self, data, weight, norm):
        return -tf.reduce_sum(weight * tf.math.log(self.Amp(data) / norm[0]))
