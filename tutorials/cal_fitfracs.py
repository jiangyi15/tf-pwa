import os.path
import re
import sys

import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir + "/..")

# import tf_pwa
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.utils import error_print, save_frac_csv


def main():
    """Calculate fit fractions and their errors."""
    import argparse

    parser = argparse.ArgumentParser(
        description="calculate fit fractions and their errors"
    )
    parser.add_argument(
        "--params", default="final_params.json", dest="params_file"
    )
    results = parser.parse_args()

    cal_fitfractions(params_file=results.params_file)


def cal_fitfractions(params_file):
    config = ConfigLoader("config.yml")
    config.set_params(params_file)
    params = config.get_params()
    # get_params_error can be replaced by
    # config.inv_he = np.load("error_matrix.npy")
    config.get_params_error(params)

    # use the file of PhaseSpace MC without efficiency indicated in config.yml
    mcdata = config.get_phsp_noeff()
    fit_frac, err_frac = config.cal_fitfractions(mcdata=mcdata, method="new")
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
    print(
        "########## fit fractions table:"
    )  # print the fit-fractions as a 2-D table. The codes below are just to implement the print function.
    print_frac_table(fit_frac)
    save_frac_csv("fit_frac.csv", fit_frac)
    save_frac_csv("fit_frac_err.csv", err_frac)


def print_frac_table(frac_dic):
    idx = [i for i in frac_dic if not isinstance(i, tuple)]
    if "sum_diag" in idx:
        idx.remove("sum_diag")

    table = []
    for i in idx:
        tmp = []
        for j in idx:
            if i == j:
                v = frac_dic[i]
            else:
                v = frac_dic.get((i, j), None)
            if v is None:
                tmp.append(" -----")
            else:
                tmp.append("{: .3f}".format(v))
        table.append(tmp)

    for i, k in zip(idx, table):
        print(i, end="\t")
        for v in k:
            print(v, end="\t")
        print()
    # the sum of all elements in the table, which should be one but for precision
    print(
        "Total sum:",
        np.sum(list(frac_dic.values())) - frac_dic.get("sum_diag", 0),
    )
    # the sum of all fit-fractions without the interference terms. We expect it to be near one.
    print(
        "Non-interference sum:",
        frac_dic.get("sum_diag", np.sum([frac_dic[i] for i in idx])),
    )


if __name__ == "__main__":
    main()
