#!/usr/bin/env python3
import sys
import os.path
import copy
from pprint import pprint
import yaml
this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir + '/..')


from tf_pwa.significance import significance
from tf_pwa.config_loader import ConfigLoader


def single_fit(config_dict):
    config = ConfigLoader(config_dict)
    
    print("\n########### initial parameters")
    pprint(config.get_params())
    print(config.full_decay)
    fit_result = config.fit()
    pprint(fit_result.params)
    # fit_result.save_as("final_params.json")
    return fit_result.min_nll, fit_result.ndf


def multi_fit(config, num=1):
    nll, ndf = single_fit(config)
    for i in range(num):
        nll_i, ndf_i = single_fit(config)
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

def cal_significance(config_name, res):
    with open(config_name) as f:
        config = yaml.safe_load(f)
    nll, ndf = single_fit(config)
    print("nll: {}, ndf: {}".format(nll, ndf))
    signi = {}
    for i in res:
        print("#veto {}".format(i))
        config_i = veto_resonance(config, i)
        nll_i, ndf_i = single_fit(config_i)
        print("nll: {}, ndf: {}".format(nll_i, ndf_i))
        signi[i] = significance(nll, nll_i, abs(ndf - ndf_i))
    return signi


def main():
    res = ["X(3940)(1+)", "Psi(4660)", "Psi(4230)", "Psi(4390)", "Psi(4260)", "Psi(4360)"]
    signi = cal_significance("config.yml", res)
    pprint(signi)


if __name__ == "__main__":
    main()
