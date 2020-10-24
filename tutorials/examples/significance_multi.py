#!/usr/bin/env python3
import sys
import os.path
import copy
from pprint import pprint
import yaml

from tf_pwa.significance import significance
from tf_pwa.config_loader import ConfigLoader, MultiConfig


def single_fit(configs, res, data):
    config = MultiConfig(configs, share_dict=res)

    print("\n########### initial parameters")
    pprint(config.get_params())
    for i in config.configs:
        print("decay chain:")
        for j in i.full_decay:
            print(j)
    fit_result = config.fit(data)
    pprint(fit_result.params)
    # fit_result.save_as("final_params.json")
    return fit_result.min_nll, fit_result.ndf, fit_result.success


def multi_fit(configs, res, data, num=5):
    nll, ndf, status = single_fit(configs, res, data)
    for i in range(num - 1):
        nll_i, ndf_i, status_i = single_fit(configs, res, data)
        assert ndf_i == ndf
        if (not status) and status_i:
            nll = nll_i
            status = status_i
        else:
            if nll > nll_i:
                nll = nll_i
    return nll, ndf


def cached_data(config_dict):
    config = ConfigLoader(config_dict)
    data = config.get_all_data()
    return data


def load_config(f_name):
    with open(f_name) as f:
        ret = yaml.safe_load(f)
    return ret


def cal_significance(config_files, res_file_name, base_res, test_res, loop=5):
    configs = [load_config(i) for i in config_files]
    res_file = load_config(res_file_name)

    for i in res_file:
        res_file[i]["disable"] = True

    data = [cached_data(i) for i in configs]

    def get_config(r):
        tmp_res = copy.deepcopy(res_file)
        for i in base_res:
            tmp_res[i]["disable"] = False
        for i in r:
            if i in base_res:
                tmp_res[i]["disable"] = True
            else:
                tmp_res[i]["disable"] = False
        return {res_file_name: tmp_res}

    base_config = get_config([])
    nll, ndf = multi_fit(configs, base_config, data, num=loop)
    nlls = {"base": nll}
    ndfs = {"base": ndf}
    print("nll: {}, ndf: {}".format(nll, ndf))
    signi = {}
    for i in test_res:
        print("\ncalculate significance for {}\n".format(i))
        config_i = get_config([i])

        nll_i, ndf_i = multi_fit(configs, config_i, data)
        nlls[i] = nll_i
        ndfs[i] = ndf_i
        signi[i] = significance(nll, nll_i, abs(ndf - ndf_i))
        print(
            "nll: {}, ndf: {}, significane: {}".format(nll_i, ndf_i, signi[i])
        )
    return signi, nlls, ndfs


def main():
    res = [
        "X(3940)(1+)",
        "X(3940)(1-)",
        "X(3940)(0-)",
        "X(3940)(2+)",
        "X(3940)(2-)",
    ]

    config_files = [
        "config.yml",
        "config_1.yml",
        "config_K3pi_1.yml",
        "config_K3pi_2.yml",
    ]
    res_files = "Resonances_B.yml"

    base_res = [
        "Psi(4660)",
        "Psi(4230)",
        "Psi(4390)",
        "Psi(4260)",
        "Psi(4360)",
    ]

    test_res = base_res  # [""]

    signi, nlls, ndfs = cal_significance(
        config_files, res_files, base_res, test_res, loop=5
    )
    print("base", nlls["base"], ndfs["base"])
    print("particle\tsignificance\tnll\tndf")
    for i in signi:
        print(i, signi[i], nlls[i], ndfs[i])


if __name__ == "__main__":
    main()
