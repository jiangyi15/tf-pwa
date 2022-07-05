import abc
import time
from typing import Any


class BaseGenerator(metaclass=abc.ABCMeta):
    DataType = Any

    @abc.abstractmethod
    def generate(self, N: int) -> BaseGenerator.DataType:
        raise NotImplementedError("generate")


class GenTest:
    def __init__(self, N_max):
        self.N_max = N_max
        self.N_gen = 0
        self.N_total = 0
        self.eff = 0.9

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
