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
        nll = self.eval_nll_end(nll, int_mc)
        return nll

    def eval_normal_factors(self, mcdata, weight=None):
        return []

    def eval_nll_part(self, data, weight=None, norm=None, idx=0):
        raise NotImplementedError("")

    def eval_nll_end(self, nll, norm=None):
        return nll

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
        a, b = ia
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
        a, b = ia
        int_mc = SumVar(int_mc_x, int_mc_g, self.Amp.trainable_variables)
        vm = self.Amp.vm
        per_replica_losses = vm.strategy.run(
            self.value_and_grad(
                lambda i0, i1: self.eval_nll_part(i0, i1, int_mc(), idx)
            ),
            args=(a, b),
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
        nll = None
        for idx, (i, j) in enumerate(zip(data, weight)):
            a, grads = self._fast_nll_part_grad((i, j), int_mc, idx)
            tmp = SumVar(a, grads, all_var)
            if nll is None:
                nll = tmp
            else:
                nll = nll + tmp
        ret, ret_grad = self.value_and_grad(
            lambda: self.eval_nll_end(nll(), int_mc())
        )()
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
        nll = None
        for idx, (i, j) in enumerate(
            zip(
                split_generator(data, batch_size=batch),
                split_generator(weight, batch_size=batch),
            )
        ):

            if int_mc is None:
                SumVar.from_call_with_hess(
                    lambda: self.eval_nll_part(i, j, None), all_var
                )
            else:
                tmp = SumVar.from_call_with_hess(
                    lambda: self.eval_nll_part(i, j, int_mc()), all_var
                )
            if nll is None:
                nll = tmp
            else:
                nll = nll + tmp
        tmp = SumVar.from_call_with_hess(
            lambda: self.eval_nll_end(nll(), int_mc()), all_var
        )
        ret, ret_grad, ret_hess = tmp.value, tmp.grad, tmp.hess
        return ret, ret_grad, ret_hess


class BaseCustomModel2(BaseCustomModel):
    function_order = [
        {
            "output": "norm",
            "function": "eval_normal_factors",
            "input": ["mcdata", "mc_weight"],
        },
        {
            "output": "nll",
            "function": "eval_nll_part",
            "input": ["data", "weight", "norm", "idx"],
        },
        {
            "output": "ret",
            "function": "eval_nll_end",
            "input": ["nll", "norm"],
        },
    ]

    def nll_grad_batch(self, data, mcdata, weight, mc_weight):
        all_var = self.Amp.trainable_variables
        n_var = len(all_var)
        zeros = np.zeros((n_var,))
        inter_data = {}
        input_data = {
            "data": data,
            "weight": weight,
            "mcdata": mcdata,
            "mc_weight": mc_weight,
        }
        ret_value = self.function_order[-1]
        for funcs in self.function_order:
            f = getattr(self, funcs["function"])
            cur_data = [
                input_data[i] for i in funcs["input"] if i in input_data
            ]
            sum_var = None
            if cur_data:
                for idx, data_i in enumerate(zip(*cur_data)):
                    cur_inter_data = {
                        k: inter_data[k]
                        for k in funcs["input"]
                        if k in inter_data
                    }
                    if "idx" in funcs["input"]:
                        tmp = f(*data_i, **cur_inter_data, idx=idx)
                    else:
                        tmp = f(*data_i, **cur_inter_data)
                    tmp = SumVar(
                        tmp,
                        tf.nest.map_structure(lambda x: zeros, tmp),
                        all_var,
                    )
                    if sum_var is None:
                        sum_var = tmp
                    else:
                        sum_var = sum_var + tmp

                inter_data[funcs["output"]] = sum_var.value
            else:
                cur_inter_data = {
                    k: inter_data[k] for k in funcs["input"] if k in inter_data
                }
                tmp = f(**cur_inter_data)
                inter_data[funcs["output"]] = tmp
        ret = inter_data[ret_value["output"]]
        grads = {
            ret_value["output"]: SumVar(
                inter_data[ret_value["output"]],
                tf.ones_like(inter_data[ret_value["output"]]),
                all_var,
            )
        }
        ret_grads = zeros
        for funcs in self.function_order[::-1]:
            f = getattr(self, funcs["function"])
            cur_data = [
                input_data[i] for i in funcs["input"] if i in input_data
            ]
            sum_var = None
            if cur_data:
                for idx, data_i in enumerate(zip(*cur_data)):
                    cur_inter_data = {
                        k: inter_data[k]
                        for k in funcs["input"]
                        if k in inter_data
                    }
                    with tf.GradientTape() as tape:
                        tf.nest.map_structure(tape.watch, cur_inter_data)
                        if "idx" in funcs["input"]:
                            tmp = f(*data_i, **cur_inter_data, idx=idx)
                        else:
                            tmp = f(*data_i, **cur_inter_data)
                        tmp = tf.nest.flatten(tmp)
                    grad_1, grad_2 = tape.gradient(
                        tmp,
                        [all_var, cur_inter_data],
                        output_gradients=grads[funcs["output"]].grad,
                        unconnected_gradients="zero",
                    )
                    ret_grads = ret_grads + tf.stack(grad_1)
                    for k, v in grad_2.items():
                        dy = tf.nest.map_structure(
                            lambda x: tf.stack(x), grad_2[k]
                        )
                        tmp = SumVar(cur_inter_data[k], dy, all_var)
                        if k not in grads:
                            grads[k] = tmp
                        else:
                            grads[k] = grads[k] + tmp
            else:
                cur_inter_data = {
                    k: inter_data[k] for k in funcs["input"] if k in inter_data
                }
                with tf.GradientTape() as tape:
                    tf.nest.map_structure(tape.watch, cur_inter_data)
                    tmp = f(**cur_inter_data)
                    tmp = tf.nest.flatten(tmp)
                grad_1, grad_2 = tape.gradient(
                    tmp,
                    [all_var, cur_inter_data],
                    output_gradients=grads[funcs["output"]].grad,
                    unconnected_gradients="zero",
                )
                ret_grads = ret_grads + tf.stack(grad_1)
                for k, v in grad_2.items():
                    dy = tf.nest.map_structure(
                        lambda x: tf.stack(x), grad_2[k]
                    )
                    tmp = SumVar(cur_inter_data[k], dy, all_var)
                    if k not in grads:
                        grads[k] = tmp
                    else:
                        grads[k] = grads[k] + tmp
        # print({k: v.grad for k, v in grads.items()})
        return ret, ret_grads


@register_nll_model("simple")
class SimpleNllModel(BaseCustomModel):
    def eval_normal_factors(self, mcdata, weight):
        return [tf.reduce_sum(self.Amp(mcdata) * weight)]

    def eval_nll_part(self, data, weight, norm, idx=0):
        nll = -tf.reduce_sum(weight * tf.math.log(self.Amp(data)))
        nll_norm = tf.reduce_sum(weight) * tf.math.log(norm[0])
        return nll + nll_norm


@register_nll_model("simple_clip")
class SimpleClipNllModel(SimpleNllModel):
    def eval_nll_part(self, data, weight, norm, idx=0):
        from .model import clip_log

        nll = -tf.reduce_sum(weight * clip_log(self.Amp(data)))
        nll_norm = tf.reduce_sum(weight) * clip_log(norm[0])
        return nll + nll_norm


@register_nll_model("simple_cfit")
class SimpleCFitModel(BaseCustomModel):
    required_params = ["bg_frac"]

    def eval_normal_factors(self, mcdata, weight):
        amp = self.Amp(mcdata) * weight * mcdata.get("eff_value", 1.0)
        a = tf.reduce_sum(amp)

        bg = weight * mcdata.get("bg_value", 1.0)
        b = tf.reduce_sum(bg)
        return [a, b]

    def eval_nll_part(self, data, weight, norm, idx=0):
        bg_frac = self.bg_frac
        pdf = (1 - bg_frac) * self.Amp(data) * data.get(
            "eff_value", 1.0
        ) / norm[0] + bg_frac * data.get("bg_value", 1.0) / norm[1]
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


def create_histogram(binning, weight, n_bins):
    index = tf.range(binning.shape[0])
    idx = tf.stack([binning, index], axis=-1)
    idx = tf.cast(idx, tf.int64)
    m = tf.sparse.SparseTensor(idx, weight, (n_bins, binning.shape[0]))
    return tf.sparse.reduce_sum(m, axis=-1)


@register_nll_model("binning_chi2")
class SimpleBinnningChi2Model(BaseCustomModel2):
    required_params = ["n_bins"]

    def eval_normal_factors(self, mcdata, weight):
        binning = mcdata.get("bin_index")
        w = self.Amp(mcdata) * weight
        ret = create_histogram(binning, w, self.n_bins)
        return tf.unstack(ret)

    def eval_nll_part(self, data, weight, norm=None, idx=0):
        binning = data.get("bin_index")
        w = weight
        ret = create_histogram(binning, w, self.n_bins)
        ret_w2 = create_histogram(binning, w**2, self.n_bins)
        return tf.unstack(ret), tf.unstack(ret_w2)

    def eval_nll_end(self, nll, norm):
        f_data, f_e = nll
        f_data = tf.stack(f_data)
        f_e = tf.stack(f_e)
        f_exp = tf.stack(norm)
        cut = f_e <= 0
        f_e = tf.where(cut, tf.ones_like(f_e), f_e)
        chi = (f_data - f_exp) ** 2 / f_e
        chi = tf.where(cut, tf.zeros_like(chi), chi)
        return tf.reduce_sum(chi) / 2


@register_nll_model("constr_frac")
class SimpleNllFracModel(BaseCustomModel):
    required_params = [
        "constr_frac"
    ]  # {name: {"res":[name], mask_params: {}, "value": 0.1, "sigma": 0.01}}

    def eval_normal_factors(self, mcdata, weight):
        int_mc = tf.reduce_sum(self.Amp(mcdata) * weight)
        ret = [int_mc]
        if self.constr_frac is None:
            self.constr_frac = {}
        for k, v in self.constr_frac.items():
            res = v.get("res", k)
            mask_params = v.get("mask_params", {})
            with self.Amp.temp_used_res(res):
                with self.Amp.mask_params(mask_params):
                    int_i = tf.reduce_sum(self.Amp(mcdata) * weight)
            ret.append(int_i)
        return ret

    def eval_nll_part(self, data, weight, norm, idx=0):
        nll = -tf.reduce_sum(weight * tf.math.log(self.Amp(data)))
        nll_norm = tf.reduce_sum(weight) * tf.math.log(norm[0])
        ret = nll + nll_norm
        if idx == 0:
            for i, (k, v) in enumerate(self.constr_frac.items()):
                value = v["value"]
                sigma = v["sigma"]
                ret = (
                    ret + 0.5 * ((norm[i + 1] / norm[0] - value) / sigma) ** 2
                )
        return ret


@register_nll_model("cfit_constr_frac")
class SimpleNllFracModel(BaseCustomModel):
    required_params = [
        "constr_frac",
        "bg_frac",
    ]  # {name: {"res":[name], mask_params: {}, "value": 0.1, "sigma": 0.01}}

    def eval_normal_factors(self, mcdata, weight):
        int_mc = tf.reduce_sum(
            self.Amp(mcdata) * mcdata.get("eff_value", 1.0) * weight
        )
        ret = [int_mc]
        if self.constr_frac is None:
            self.constr_frac = {}
        for k, v in self.constr_frac.items():
            res = v.get("res", k)
            mask_params = v.get("mask_params", {})
            with self.Amp.temp_used_res(res):
                with self.Amp.mask_params(mask_params):
                    int_i = tf.reduce_sum(self.Amp(mcdata) * weight)
            ret.append(int_i)
        bg = weight * mcdata.get("bg_value", 1.0)
        b = tf.reduce_sum(bg)
        ret.append(b)
        return ret

    def eval_nll_part(self, data, weight, norm, idx=0):
        bg_frac = self.bg_frac
        pdf = (1 - bg_frac) * self.Amp(data) * data.get(
            "eff_value", 1.0
        ) / norm[0] + bg_frac * data.get("bg_value", 1.0) / norm[-1]
        ret = -tf.reduce_sum(weight * tf.math.log(pdf))
        if idx == 0:
            for i, (k, v) in enumerate(self.constr_frac.items()):
                value = v["value"]
                sigma = v["sigma"]
                ret = (
                    ret + 0.5 * ((norm[i + 1] / norm[0] - value) / sigma) ** 2
                )
        return ret
