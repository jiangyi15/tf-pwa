import numpy as np
import tensorflow as tf

from tf_pwa.data import split_generator
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
        int_mc = self.eval_normal_factors(mcdata, mc_weight)
        nll = self.eval_nll_part(data, weight, int_mc, idx=0)
        return nll

    def eval_normal_factors(self, mcdata, weight=None):
        return []

    def eval_nll_part(self, data, weight=None, norm=None, idx=0):
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
            tmp = SumVar(a, [tf.stack(i) for i in grads], all_var)
            if int_mc is None:
                int_mc = tmp
            else:
                int_mc = int_mc + tmp
        ret = 0.0
        ret_grad = 0.0
        for idx, (i, j) in enumerate(zip(data, weight)):
            with tf.GradientTape() as tape:
                if int_mc is None:
                    a = self.eval_nll_part(i, j, None, idx=idx)
                else:
                    a = self.eval_nll_part(i, j, int_mc(), idx=idx)
            grads = tape.gradient(a, all_var, unconnected_gradients="zero")
            ret = ret + a
            ret_grad = ret_grad + tf.stack(grads)
        return ret, ret_grad

    def nll_grad_hessian(
        self, data, mcdata, weight=1.0, batch=24000, bg=None, mc_weight=1.0
    ):
        all_var = self.Amp.trainable_variables
        n_var = len(all_var)
        int_mc = None
        for i, j in zip(
            split_generator(mcdata, batch_size=batch),
            split_generator(mc_weight, batch_size=batch),
        ):
            with tf.GradientTape(persistent=True) as tape0:
                with tf.GradientTape(persistent=True) as tape:
                    a = self.eval_normal_factors(i, j)
                grads = [
                    tape.gradient(ai, all_var, unconnected_gradients="zero")
                    for ai in a
                ]
                del tape
            hess = []
            for gi in grads:
                tmp = []
                for gij in gi:
                    tmp.append(
                        tape0.gradient(
                            gij, all_var, unconnected_gradients="zero"
                        )
                    )
                hess.append(tmp)
            del tape0
            hess = [tf.stack(i) for i in hess]
            tmp = SumVar(a, [tf.stack(i) for i in grads], all_var, hess=hess)
            if int_mc is None:
                int_mc = tmp
            else:
                int_mc = int_mc + tmp
        ret = 0.0
        ret_grad = 0.0
        ret_hess = 0.0
        for idx, (i, j) in enumerate(
            zip(
                split_generator(data, batch_size=batch),
                split_generator(weight, batch_size=batch),
            )
        ):
            with tf.GradientTape(persistent=True) as tape0:
                with tf.GradientTape() as tape:
                    if int_mc is None:
                        a = self.eval_nll_part(i, j, None, idx=idx)
                    else:
                        a = self.eval_nll_part(i, j, int_mc(), idx=idx)
                grads = tape.gradient(a, all_var, unconnected_gradients="zero")
            hess = [
                tape0.gradient(gi, all_var, unconnected_gradients="zero")
                for gi in grads
            ]
            del tape0
            ret = ret + a
            ret_grad = ret_grad + tf.stack(grads)
            ret_hess = ret_hess + tf.stack(hess)
        return ret, ret_grad, ret_hess


@register_nll_model("simple")
class SimpleNllModel(BaseCustomModel):
    def eval_normal_factors(self, mcdata, weight):
        return [tf.reduce_sum(self.Amp(mcdata) * weight)]

    def eval_nll_part(self, data, weight, norm, idx=0):
        nll = -tf.reduce_sum(weight * tf.math.log(self.Amp(data)))
        nll_norm = tf.reduce_sum(weight) * tf.math.log(norm[0])
        return nll + nll_norm


@register_nll_model("simple_chi2")
class SimpleChi2Model(BaseCustomModel):
    """
    fit amp = weight directly. Required set extended = True.
    """

    def eval_nll_part(self, data, weight, norm, idx=0):
        nll = 0.5 * tf.reduce_sum((weight - self.Amp(data)) ** 2)
        return nll
