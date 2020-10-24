import sys
import os.path

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir + "/..")
import numpy as np
import re

# import tf_pwa
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.utils import error_print
from tf_pwa.applications import fit_fractions


def main():
    """Calculate fit fractions and their errors."""
    import argparse

    parser = argparse.ArgumentParser(
        description="calculate fit fractions and their errors"
    )
    parser.add_argument("--params", default="final_params.json", dest="params_file")
    results = parser.parse_args()

    cal_fitfractions(params_file=results.params_file)


def cal_fitfractions(params_file):
    config = ConfigLoader("config.yml")
    config.set_params(params_file)
    params = config.get_params()
    config.get_params_error(params)

    mcdata = (
        config.get_phsp_noeff()
    )  # use the file of PhaseSpace MC without efficiency indicated in config.yml
    fit_frac, err_frac = fit_fractions(
        config.get_amplitude(), mcdata, config.inv_he, params
    )
    print("########## fit fractions:")
    fit_frac_string = ""
    for i in fit_frac:
        if isinstance(i, tuple):
            name = "{}x{}".format(*i)  # interference term
        else:
            name = i  # fit fraction
        fit_frac_string += "{} {}\n".format(
            name, error_print(fit_frac[i], err_frac.get(i, None))
        )
    print(fit_frac_string)
    print("########## fit fractions table:")
    print_frac_table(
        fit_frac_string
    )  # print the fit-fractions as a 2-D table. The codes below are just to implement the print function.


def print_frac_table(frac_txt):
    def get_point(s):
        partten = re.compile(r"([^\s]+)\s+([+-.e1234567890]+)\s+")
        ret = {}
        for i in s.split("\n"):
            g = partten.match(i)
            if g:
                name = g.group(1).split("x")
                frac = float(g.group(2))
                if len(name) == 1:
                    l, r = name * 2
                elif len(name) == 2:
                    l, r = name
                else:
                    raise Exception("error {}".format(name))
                if l not in ret:
                    ret[l] = {}
                ret[l][r] = frac
        return ret

    s = get_point(frac_txt)

    def get_table(s):
        idx = list(s)
        n_idx = len(idx)
        idx_map = dict(zip(idx, range(n_idx)))
        ret = []
        for i in range(n_idx):
            ret.append([0.0 for j in range(n_idx)])
        for i, k in s.items():
            for j, v in k.items():
                ret[idx_map[i]][idx_map[j]] = v
        return idx, ret

    idx, table = get_table(s)

    ret = []
    for i, k in enumerate(table):
        tmp = []
        for j, v in enumerate(k):
            if i < j:
                tmp.append("-")
            else:
                tmp.append("{:.3f}".format(v))
        ret.append(tmp)
    for i, k in zip(idx, ret):
        print(i, end="\t")
        for v in k:
            print(v, end="\t")
        print()
    print(
        "Total sum:", np.sum(table)
    )  # the sum of all elements in the table, which should be one but for precision
    print(
        "Non-interference sum:", np.sum(np.diagonal(table))
    )  # the sum of all fit-fractions without the interference terms. We expect it to be near one.


if __name__ == "__main__":
    main()
