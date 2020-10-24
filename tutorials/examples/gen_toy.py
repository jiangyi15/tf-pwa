#!/usr/bin/env python3
import sys
import os.path
import copy
from pprint import pprint
import yaml

this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + "/..")

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.phasespace import PhaseSpaceGenerator
from tf_pwa.data import data_mask, data_index, data_to_numpy
import numpy as np
import json


def gen_phasespace(top, finals, number):
    a = PhaseSpaceGenerator(top, finals)
    flat_mc_data = a.generate(number)
    return flat_mc_data


def simple_select(phsp, amp):
    weights = amp(phsp)
    max_weights = np.max(weights) * 1.01

    rnd = np.random.random(weights.shape)
    select_index = weights / max_weights > rnd
    select_data = data_mask(phsp, select_index)
    return select_data


def save_dat(file_name, data, config):
    idx = [("particle", i, "p") for i in config.get_dat_order(True)]
    data = data_to_numpy(data)
    data_p = np.array([data_index(data, i) for i in idx])

    dat = np.transpose(data_p, (1, 0, 2)).reshape((-1, 4))
    np.savetxt(file_name, dat)
    dat = np.loadtxt(file_name)


def gen_toy(config, params, file_name):
    config = ConfigLoader(config)
    try:
        config.set_params(params)
    except Exception as e:
        print(e)

    print("using params")
    print(json.dumps(config.get_params(), indent=2))
    amp = config.get_amplitude()
    phsp = config.get_data("phsp")

    select_data = simple_select(phsp, amp)

    save_dat(file_name, select_data, config)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="calculate significance")
    parser.add_argument("--config", default="config.yml", dest="config")
    parser.add_argument("--params", default="final_params.json", dest="params")
    parser.add_argument("--output", default="data/toy.dat", dest="output")
    results = parser.parse_args()
    gen_toy(results.config, results.params, results.output)


if __name__ == "__main__":
    main()
