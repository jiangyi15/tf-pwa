import numpy as np
import tensorflow as tf

from tf_pwa.data import data_shape, split_generator
from tf_pwa.variable import SumVar

from .model import Model, register_nll_model

"""
Custom nll model
"""


class BaseCustomModel(Model):
    def value_and_grad(self, fun):
        all_var = self.Amp.trainable_variables
        n_var = len(all_var)

        def _fun(*args, **kwargs):
            with tf.GradientTape(persistent=True) as tape:
                y = fun(*args, **kwargs)
            dy = tf.nest.map_structure(
                lambda x: tf.stack(
                    tape.gradient(x, all_var, unconnected_gradients="zero")
                ),
                y,
            )
            del tape
            return y, dy

        return _fun

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

    def _fast_int_mc_grad(self, data):
        if self.Amp.vm.strategy is not None:
            return self._fast_int_mc_grad_multi(data)
        else:
            return self.value_and_grad(self.eval_normal_factors)(
                data[0], data[1]
            )

    def _fast_nll_part_grad(self, data, int_mc=None, idx=0):
        if int_mc is None:
            all_var = self.Amp.trainable_variables
            n_var = len(all_var)
            int_mc = SumVar([np.array(1.0)], [np.zeros((n_var,))], all_var)
        if self.Amp.vm.strategy is not None:
            return self._fast_nll_part_grad_multi(
                data, int_mc.value, int_mc.grad, idx
            )
        else:
            return self.value_and_grad(
                lambda: self.eval_nll_part(data[0], data[1], int_mc(), idx)
            )()

    @tf.function
    def _fast_int_mc_grad_multi(self, ia):
        strategy = self.Amp.vm.strategy
        n_p = strategy.num_replicas_in_sync
        ia = list(
            split_generator(ia, batch_size=(data_shape(ia) + n_p - 1) // n_p)
        )

        def _tmp_fun(ctx):
            return ia[ctx.replica_id_in_sync_group]

        i = strategy.experimental_distribute_values_from_function(_tmp_fun)
        a, b = i
        vm = self.Amp.vm
        per_replica_losses = vm.strategy.run(
            self.value_and_grad(self.eval_normal_factors), args=(a, b)
        )
        tmp = vm.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )
        return tmp

    @tf.function
    def _fast_nll_part_grad_multi(self, ia, int_mc_x, int_mc_g, idx):
        strategy = self.Amp.vm.strategy
        n_p = strategy.num_replicas_in_sync
        ia = list(
            split_generator(ia, batch_size=(data_shape(ia) + n_p - 1) // n_p)
        )

        def _tmp_fun(ctx):
            return ia[ctx.replica_id_in_sync_group]

        ab = strategy.experimental_distribute_values_from_function(_tmp_fun)
        int_mc = SumVar(int_mc_x, int_mc_g, self.Amp.trainable_variables)
        vm = self.Amp.vm
        per_replica_losses = vm.strategy.run(
            self.value_and_grad(
                lambda i: self.eval_nll_part(i[0], i[1], int_mc(), idx)
            ),
            args=(ab,),
        )
        tmp = vm.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )
        return tmp

    def nll_grad_batch(self, data, mcdata, weight, mc_weight):
        all_var = self.Amp.trainable_variables
        n_var = len(all_var)
        int_mc = None  # SumVar(0., np.zeros((n_var,)), all_var)
        for i, j in zip(mcdata, mc_weight):
            a, grad = self._fast_int_mc_grad((i, j))
            tmp = SumVar(a, grad, all_var)
            if int_mc is None:
                int_mc = tmp
            else:
                int_mc = int_mc + tmp
        ret = 0.0
        ret_grad = 0.0
        for idx, (i, j) in enumerate(zip(data, weight)):
            a, grads = self._fast_nll_part_grad((i, j), int_mc, idx)
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
            tmp = SumVar.from_call_with_hess(
                lambda: self.eval_normal_factors(i, j), all_var
            )
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


@register_nll_model("simple_cfit")
class SimpleNllModel(BaseCustomModel):
    def eval_normal_factors(self, mcdata, weight):
        amp = self.Amp(mcdata) * weight * mcdata.get("eff_value", 1.0)
        a = tf.reduce_sum(amp)

        bg = weight * mcdata.get("bg_value", 1.0)
        b = tf.reduce_sum(bg)
        return [a, b]

    def eval_nll_part(self, data, weight, norm, idx=0):
        bg_frac = self.w_bkg
        pdf = (1 - bg_frac) * self.Amp(data) / norm[0] + bg_frac * data.get(
            "bg_value", 1.0
        ) / norm[1]
        nll = -tf.reduce_sum(weight * tf.math.log(pdf))
        return nll


@register_nll_model("simple_chi2")
class SimpleChi2Model(BaseCustomModel):
    """
    fit amp = weight directly. Required set extended = True.
    """

    def eval_nll_part(self, data, weight, norm, idx=0):
        nll = 0.5 * tf.reduce_sum((weight - self.Amp(data)) ** 2)
        return nll
