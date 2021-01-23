"""
Scripts for calculating fit fraction of each LS component.

Need config.yml and final_params.json (and error_matrix.npy for error).

"""

import itertools

import numpy as np
import tensorflow as tf

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.data import split_generator
from tf_pwa.experimental.opt_int import split_gls
from tf_pwa.model.model import sum_gradient


def cal_each_ls(var, dg, dec_chain, data, weights):
    """
    calculate amplitude square for each L-S coupling.
    """
    cached = []
    for i, dc in split_gls(dec_chain):
        int_mc, grad = sum_gradient(dg.sum_amp, data, var, weights)
        cached.append((int_mc, grad))
    return cached


def cal_each_decay_chain(var, dec, data, weights):
    """
    calculate amplitude square for each decay chain (resonant).
    """
    hij = []
    used_chains = dec.chains_idx
    index = []
    for k, i in enumerate(dec):
        dec.set_used_chains([k])
        tmp = cal_each_ls(var, dec, i, data, weights)
        hij.append(tmp)
        index.append(i)
    dec.set_used_chains(used_chains)  # recover
    return index, hij


def cal_frac(amp, phsp, err_matrix, batch=300000):
    decay_group = amp.decay_group
    var = amp.trainable_variables

    # split large data to avoid OOM
    phsp = list(split_generator(phsp, batch))
    weights = [i.get("weight", 1.0) for i in phsp]

    # denominator part
    int_mc, grad_total = sum_gradient(amp, phsp, var, weights)
    int_mc = int_mc.numpy()
    grad_total = np.array([i.numpy() for i in grad_total])

    idx, amp_matrix2 = cal_each_decay_chain(var, decay_group, phsp, weights)

    # numerator part for each decay chain
    total = 0
    for k, v in zip(idx, amp_matrix2):
        print("decay chain:", k)
        ls_list = [i.get_ls_list() for i in k]
        ls = itertools.product(*ls_list)
        for i, (value, grad) in enumerate(v):
            # fraction gradient
            value = value.numpy()
            grad = np.array([i.numpy() for i in grad])
            grad_f = (grad - grad_total * value / int_mc) / int_mc

            # error propagation
            error = np.sqrt(np.dot(np.dot(err_matrix, grad_f), grad_f))
            print(
                f"  frac of ls: {next(ls)} is "
                f"{value/int_mc:.4f} +/- {error:.4f}"
            )
            total += value
    print("sum diagonal:", total / int_mc)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="calculate fit fractions")
    parser.add_argument("-c", "--config", default="config.yml")
    parser.add_argument("-i", "--init_params", default="final_params.json")
    parser.add_argument("-e", "--error_matrix", default="error_matrix.npy")
    results = parser.parse_args()

    # load model and parameters and error matrix
    config = ConfigLoader(results.config)
    config.set_params(results.init_params)
    err_matrix = np.load(results.error_matrix)

    amp = config.get_amplitude()
    phsp = config.get_phsp_noeff()  # get_data("phsp")[0]

    cal_frac(amp, phsp, err_matrix)


if __name__ == "__main__":
    main()
