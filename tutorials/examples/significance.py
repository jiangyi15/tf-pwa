#!/usr/bin/env python3
import sys
import os.path
import copy
from pprint import pprint
import yaml

this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + "/..")


from tf_pwa.significance import significance
from tf_pwa.config_loader import ConfigLoader


def single_fit(config_dict, data, phsp, bg):
    config = ConfigLoader(config_dict)

    print("\n########### initial parameters")
    pprint(config.get_params())
    print(config.full_decay)
    fit_result = config.fit(data, phsp, bg=bg)
    pprint(fit_result.params)
    # fit_result.save_as("final_params.json")
    return fit_result.min_nll, fit_result.ndf


def multi_fit(config, data, phsp, bg, num=5):
    nll, ndf = single_fit(config, data, phsp, bg)
    for i in range(num - 1):
        nll_i, ndf_i = single_fit(config, data, phsp, bg)
        assert ndf_i == ndf
        if nll > nll_i:
            nll = nll_i
    return nll, ndf


def veto_resonance(config, res):
    config = copy.deepcopy(config)
    for k, v in config["particle"].items():
        if isinstance(v, list):
            if res in v:
                config["particle"][k].remove(res)
    if res in config["decay"]:
        config["decay"].remove(res)
    for k, v in config["decay"].items():
        if res in v:
            decay[k].remove(res)
        for j in v:
            if isinstance(j, list):
                if res in j:
                    j.remove(res)
    return config


def cached_data(config_dict):
    config = ConfigLoader(config_dict)
    data, phsp, bg = config.get_all_data()[:3]
    return data, phsp, bg


def cal_significance(config_name, res, model="-"):
    with open(config_name) as f:
        config = yaml.safe_load(f)
    data, phsp, bg = cached_data(config)

    def get_config(extra=[]):
        base_conf = copy.deepcopy(config)
        if model == "+":
            veto_res = res.copy()
            for i in extra:
                if i in veto_res:
                    veto_res.remove(i)
        else:
            veto_res = extra
        for i in veto_res:
            base_conf = veto_resonance(base_conf, i)
        return base_conf

    base_config = get_config([])
    nll, ndf = multi_fit(base_config, data, phsp, bg)
    nlls = {"base": nll}
    ndfs = {"base": ndf}
    print("nll: {}, ndf: {}".format(nll, ndf))
    signi = {}
    for i in res:
        print("\ncalculate significance for {}\n".format(i))
        config_i = get_config([i])
        nll_i, ndf_i = multi_fit(config_i, data, phsp, bg)
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
        "Psi(4660)",
        "Psi(4230)",
        "Psi(4390)",
        "Psi(4260)",
        "Psi(4360)",
    ]
    import argparse

    parser = argparse.ArgumentParser(description="calculate significance")
    parser.add_argument("--config", default="config.yml", dest="config")
    results = parser.parse_args()
    signi, nlls, ndfs = cal_significance(results.config, res, model="+")
    print("base", nlls["base"], ndfs["base"])
    print("particle\tsignificance\tnll\tndf")
    for i in signi:
        print(i, signi[i], nlls[i], ndfs[i])


if __name__ == "__main__":
    main()
