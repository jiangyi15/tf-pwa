#!/usr/bin/env python3

import csv
import json
import time
from pprint import pprint

# avoid using Xwindow
import matplotlib

matplotlib.use("agg")

import tensorflow as tf

# examples of custom particle model
from tf_pwa.amp import simple_resonance
from tf_pwa.config_loader import ConfigLoader, MultiConfig
from tf_pwa.experimental import extra_amp, extra_data
from tf_pwa.utils import error_print, tuple_table


@simple_resonance("New", params=["alpha", "beta"])
def New_Particle(m, alpha, beta=0):
    """example Particle model define, can be used in config.yml as `model: New`"""
    zeros = tf.zeros_like(m)
    r = -tf.complex(alpha, beta) * tf.complex(m, zeros)
    return tf.exp(r)


def json_print(dic):
    """print parameters as json"""
    s = json.dumps(dic, indent=2)
    print(s, flush=True)


def load_config(config_file="config.yml", total_same=False):
    config_files = config_file.split(",")
    if len(config_files) == 1:
        return ConfigLoader(config_files[0])
    return MultiConfig(config_files, total_same=total_same)


def print_fit_result(config, fit_result):
    print("\n########## fit results:")
    print("Fit status: ", fit_result.success)
    print("Minimal -lnL = ", fit_result.min_nll)
    for k, v in fit_result.params.items():
        print(k, error_print(v, fit_result.error.get(k, None)))


def print_fit_result_roofit(config, fit_result):
    import numpy as np

    value = fit_result.params
    params_name = config.vm.trainable_vars
    n_par = len(params_name)
    name_size = max(len(i) for i in params_name)
    # fcn = config.get_fcn()
    # _, grad = fcn.nll_grad(fit_result.params)
    # edm = np.dot(np.dot(config.inv_he, grad), grad)
    print(
        "FCN={:.1f} FROM HESSE \t STATUS={}".format(
            fit_result.min_nll, "OK" if fit_result.success else "FAILED"
        )
    )
    # print("    \t EDM={:.6e}".format(edm))
    print(
        " NO. \t{:<n_size} \t VALUE  \t ERROR".replace(
            "n_size", str(name_size)
        ).format("NAME")
    )
    for i, name in enumerate(params_name):
        s = " {:>3} \t{:<n_size} \t{: .6e} \t{: .6e}".replace(
            "n_size", str(name_size)
        ).format(
            i, name, fit_result.params[name], fit_result.error.get(name, 0.0)
        )
        print(s)
    print("\nERROR MATRIX.   NPAR={}".format(n_par))
    for i in range(n_par):
        for j in range(n_par):
            print("{: .3e} ".format(config.inv_he[i, j]), end="")
        print("")
    print("\nPARAMETER CORRELATION COEFFICIENTS")
    print(
        " NO.  \tGLOBAL   "
        + " ".join(["{:<10}".format(i) for i in range(n_par)])
    )
    err = np.sqrt(np.diag(config.inv_he))
    correlation = config.inv_he / err[None, :] / err[:, None]
    inv_correlation = np.linalg.inv(correlation)
    for i in range(n_par):
        print(" {:>3}  \t".format(i), end="")
        dom = inv_correlation[i, i] * correlation[i, i]
        print(
            "{:.4f} ".format(
                np.where((dom < 0.0) | (dom > 1.0), 0.0, np.sqrt(1 - 1 / dom))
            ),
            end="",
        )
        for j in range(n_par):
            print("{: .3e} ".format(correlation[i, j]), end="")
        print("")


def fit(
    config,
    init_params="",
    method="BFGS",
    loop=1,
    maxiter=500,
    printer="roofit",
):
    """
    simple fit script
    """
    # load config.yml
    # config = ConfigLoader(config_file)

    # load data
    all_data = config.get_all_data()

    fit_results = []
    for i in range(loop):
        # set initial parameters if have
        if config.set_params(init_params):
            print("using {}".format(init_params))
        else:
            print("\nusing RANDOM parameters", flush=True)
        # try to fit
        try:
            fit_result = config.fit(
                batch=65000, method=method, maxiter=maxiter
            )
        except KeyboardInterrupt:
            config.save_params("break_params.json")
            raise
        except Exception as e:
            print(e)
            config.save_params("break_params.json")
            raise
        fit_results.append(fit_result)
        # reset parameters
        try:
            config.reinit_params()
        except Exception as e:
            print(e)

    fit_result = fit_results.pop()
    for i in fit_results:
        if i.success:
            if not fit_result.success or fit_result.min_nll > i.min_nll:
                fit_result = i

    config.set_params(fit_result.params)
    json_print(fit_result.params)
    fit_result.save_as("final_params.json")

    # calculate parameters error
    if maxiter != 0:
        fit_error = config.get_params_error(fit_result, batch=13000)
        fit_result.set_error(fit_error)
        fit_result.save_as("final_params.json")
        pprint(fit_error)
        print("\n########## fit results:")
        if printer == "roofit":
            print_fit_result_roofit(config, fit_result)
        else:
            print_fit_result(config, fit_result)

    return fit_result


def write_some_results(config, fit_result, save_root=False):
    # plot partial wave distribution
    config.plot_partial_wave(fit_result, plot_pull=True, save_root=save_root)

    # calculate fit fractions
    phsp_noeff = config.get_phsp_noeff()
    fit_frac, err_frac = config.cal_fitfractions({}, phsp_noeff)

    print("########## fit fractions")
    fit_frac_string = ""
    for i in fit_frac:
        if isinstance(i, tuple):
            name = "{}x{}".format(*i)
        else:
            name = i
        fit_frac_string += "{} {}\n".format(
            name, error_print(fit_frac[i], err_frac.get(i, None))
        )
    print(fit_frac_string)
    save_frac_csv("fit_frac.csv", fit_frac)
    save_frac_csv("fit_frac_err.csv", err_frac)
    # from frac_table import frac_table
    # frac_table(fit_frac_string)
    # chi2, ndf = config.cal_chi2(mass=["R_BC", "R_CD"], bins=[[2,2]]*4)


def write_some_results_combine(config, fit_result, save_root=False):

    from tf_pwa.applications import fit_fractions

    for i, c in enumerate(config.configs):
        c.plot_partial_wave(
            fit_result, prefix="figure/s{}_".format(i), save_root=save_root
        )

    for it, config_i in enumerate(config.configs):
        print("########## fit fractions {}:".format(it))
        mcdata = config_i.get_phsp_noeff()
        fit_frac, err_frac = fit_fractions(
            config_i.get_amplitude(),
            mcdata,
            config.inv_he,
            fit_result.params,
        )
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
        save_frac_csv(f"fit_frac{it}.csv", fit_frac)
        save_frac_csv(f"fit_frac{it}_err.csv", err_frac)
    # from frac_table import frac_table
    # frac_table(fit_frac_string)


def save_frac_csv(file_name, fit_frac):
    table = tuple_table(fit_frac)
    with open(file_name, "w") as f:
        f_csv = csv.writer(f)
        f_csv.writerows(table)


def write_run_point():
    """write time as a point of fit start"""
    with open(".run_start", "w") as f:
        localtime = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(time.time())
        )
        f.write(localtime)


def main():
    """entry point of fit. add some arguments in commond line"""
    import argparse

    parser = argparse.ArgumentParser(description="simple fit scripts")
    parser.add_argument(
        "--no-GPU", action="store_false", default=True, dest="has_gpu"
    )
    parser.add_argument("-c", "--config", default="config.yml", dest="config")
    parser.add_argument(
        "-i", "--init_params", default="init_params.json", dest="init"
    )
    parser.add_argument("-m", "--method", default="BFGS", dest="method")
    parser.add_argument("-l", "--loop", type=int, default=1, dest="loop")
    parser.add_argument(
        "-x", "--maxiter", type=int, default=2000, dest="maxiter"
    )
    parser.add_argument("-r", "--save_root", default=False, dest="save_root")
    parser.add_argument(
        "--total-same", action="store_true", default=False, dest="total_same"
    )
    parser.add_argument("--printer", default="roofit", dest="printer")
    results = parser.parse_args()
    if results.has_gpu:
        devices = "/device:GPU:0"
    else:
        devices = "/device:CPU:0"
    with tf.device(devices):
        config = load_config(results.config, results.total_same)
        fit_result = fit(
            config,
            results.init,
            results.method,
            results.loop,
            results.maxiter,
            results.printer,
        )
        if isinstance(config, ConfigLoader):
            write_some_results(config, fit_result, save_root=results.save_root)
        else:
            write_some_results_combine(
                config, fit_result, save_root=results.save_root
            )


if __name__ == "__main__":
    write_run_point()
    main()
