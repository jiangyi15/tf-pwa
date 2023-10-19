import numpy as np
import tensorflow as tf

from tf_pwa.data import split_generator
from tf_pwa.model import Model, register_nll_model
from tf_pwa.variable import SumVar

"""
Custom nll model
"""


def deep_stack(dic, deep=1):
    if isinstance(dic, dict):
        return {k: v for k, v in dic.items()}
    elif isinstance(dic, list):
        flag = True
        tmp = dic
        for i in range(deep):
            if len(tmp) > 0 and isinstance(tmp, list):
                tmp = tmp[0]
            else:
                flag = False
                break
        if flag and isinstance(tmp, tf.Tensor):
            return tf.stack(dic)
        else:
            return [deep_stack(i, deep) for i in dic]
    elif isinstance(dic, tuple):
        return tuple([deep_stack(i, deep) for i in dic])
    return dic


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
            grads = tf.nest.map_structure(
                lambda x: tf.stack(
                    tape.gradient(x, all_var, unconnected_gradients="zero")
                ),
                a,
            )
            tmp = SumVar(a, grads, all_var)
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
                grads = tf.nest.map_structure(
                    lambda x: tape.gradient(
                        x, all_var, unconnected_gradients="zero"
                    ),
                    a,
                )
                del tape
            hess = tf.nest.map_structure(
                lambda x: tf.stack(
                    tape0.gradient(x, all_var, unconnected_gradients="zero")
                ),
                grads,
            )
            del tape0
            tmp = SumVar(a, deep_stack(grads), all_var, hess=deep_stack(hess))
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
