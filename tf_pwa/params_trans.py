import contextlib

import numpy as np
import tensorflow as tf


class ParamsTrans:
    def __init__(self, vm, err_matrix):
        self.vm = vm
        self.err_matrix = err_matrix
        self.tape = None

    @contextlib.contextmanager
    def trans(self):
        with tf.GradientTape(persistent=True) as tape:
            yield self
        self.tape = tape

    def get_grad(self, val, keep=False):
        grad = self.tape.gradient(
            val,
            self.vm.trainable_variables,
            unconnected_gradients="zero",
        )
        print(grad)
        grad = tf.stack(grad, axis=-1)
        if not keep:
            del self.tape
        return grad

    def get_error(self, vals, keep=False):
        if isinstance(vals, (list, tuple)):
            ret = type(vals)([self.get_error(v, keep=True) for v in vals])
        elif isinstance(vals, dict):
            ret = {k: self.get_error(v, keep=True) for k, v in vals.items()}
        elif isinstance(vals, tf.Tensor):
            if len(vals.shape) == 0:  # scalar
                grad = self.tape.gradient(
                    vals,
                    self.vm.trainable_variables,
                    unconnected_gradients="zero",
                )
                grad = tf.stack(grad, axis=-1)
                ret = tf.sqrt(
                    tf.reduce_sum(
                        tf.linalg.matvec(self.err_matrix, grad) * grad
                    )
                )
            else:  # vector
                grad = self.tape.jacobian(
                    vals,
                    self.vm.trainable_variables,
                    unconnected_gradients="zero",
                )
                grad = tf.stack(grad, axis=-1)
                grad = tf.reshape(grad, (-1, len(self.vm.trainable_variables)))
                new_err_matrix = tf.matmul(
                    tf.matmul(grad, self.err_matrix), grad, transpose_b=True
                )
                ret = tf.sqrt(tf.linalg.tensor_diag_part(new_err_matrix))
                ret = tf.reshape(ret, vals.shape)
        else:
            raise TypeError(
                f"unsuported type {type(vals)}, use tensor instead"
            )
        if not keep:
            del self.tape
        return ret

    def get_error_matrix(self, vals, keep=False):
        if isinstance(vals, (list, tuple)):
            grad = [
                self.tape.gradient(
                    i,
                    self.vm.trainable_variables,
                    unconnected_gradients="zero",
                )
                for i in vals
            ]
        elif isinstance(vals, tf.Tensor):
            grad = self.tape.jacobian(
                vals, self.vm.trainable_variables, unconnected_gradients="zero"
            )
        if not keep:
            del self.tape
        # print(grad)
        grad = np.stack(grad).reshape((-1, len(self.vm.trainable_variables)))
        # print(grad, self.err_matrix, np.dot(grad, self.err_matrix), grad.T)
        return np.dot(np.dot(grad, self.err_matrix), grad.T)

    def __getitem__(self, key):
        return self.vm.variables[key]
