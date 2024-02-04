import abc
import time
from typing import Any


class BaseGenerator(metaclass=abc.ABCMeta):
    DataType = Any

    @abc.abstractmethod
    def generate(self, N: int) -> Any:
        raise NotImplementedError("generate")


class GenTest:
    def __init__(self, N_max, display=True):
        self.N_max = N_max
        self.N_gen = 0
        self.N_total = 0
        self.eff = 0.9
        self.display = display

    def generate(self, N):
        self.N_gen = 0
        self.N_total = 0

        N_progress = 50
        start_time = time.perf_counter()
        while self.N_gen < N:
            test_N = min(int((N - self.N_gen) / self.eff * 1.1), self.N_max)
            self.N_total += test_N
            yield test_N
            progress = self.N_gen / N + 1e-5
            finsh = "▓" * int(progress * N_progress)
            need_do = "-" * (N_progress - int(progress * N_progress) - 1)
            now = time.perf_counter() - start_time
            if self.display:
                print(
                    "\r{:^3.1f}%[{}>{}] {:.2f}/{:.2f}s eff: {:.6f}%  ".format(
                        progress * 100,
                        finsh,
                        need_do,
                        now,
                        now / progress,
                        self.eff * 100,
                    ),
                    end="",
                )
            self.eff = (self.N_gen + 1) / (self.N_total + 1)  # avoid zero
        end_time = time.perf_counter() - start_time
        if self.display:
            print(
                "\r{:^3.1f}%[{}] {:.2f}/{:.2f}s  eff: {:.6f}%   ".format(
                    100, "▓" * N_progress, end_time, end_time, self.eff * 100
                )
            )

    def add_gen(self, n_gen):
        # print("add gen")
        self.N_gen = self.N_gen + n_gen

    def set_gen(self, n_gen):
        # print("set gen")
        self.N_gen = n_gen


def multi_sampling(
    phsp,
    amp,
    N,
    max_N=200000,
    force=True,
    max_weight=None,
    importance_f=None,
    display=True,
):

    import tensorflow as tf

    from tf_pwa.data import data_mask, data_merge, data_shape

    a = GenTest(max_N, display=display)
    all_data = []

    for i in a.generate(N):
        data, new_max_weight = single_sampling2(
            phsp, amp, i, max_weight, importance_f
        )
        if max_weight is None:
            max_weight = new_max_weight * 1.1
        if new_max_weight > max_weight and len(all_data) > 0:
            tmp = data_merge(*all_data)
            rnd = tf.random.uniform((data_shape(tmp),), dtype=max_weight.dtype)
            cut = (
                rnd * new_max_weight / max_weight < 1.0
            )  # .max_amplitude < 1.0
            max_weight = new_max_weight * 1.05
            tmp = data_mask(tmp, cut)
            all_data = [tmp]
            a.set_gen(data_shape(tmp))
        a.add_gen(data_shape(data))
        # print(a.eff, a.N_gen, max_weight)
        all_data.append(data)

    ret = data_merge(*all_data)

    if force:
        cut = tf.range(data_shape(ret)) < N
        ret = data_mask(ret, cut)

    status = (a, max_weight)

    return ret, status


def single_sampling2(phsp, amp, N, max_weight=None, importance_f=None):
    import tensorflow as tf

    from tf_pwa.data import data_mask

    data = phsp(N)
    weight = amp(data)
    if importance_f is not None:
        weight = weight / importance_f(data)
    new_max_weight = tf.reduce_max(weight)
    if max_weight is None or max_weight < new_max_weight:
        max_weight = new_max_weight * 1.01
    rnd = tf.random.uniform(weight.shape, dtype=weight.dtype)
    cut = rnd * max_weight < weight
    data = data_mask(data, cut)
    return data, max_weight
