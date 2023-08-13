import os.path
import sys

import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir + "/..")

# import tf_pwa
from tf_pwa.applications import corr_coef_matrix
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.utils import error_print


def main():
    """Calculate errors of a given set of parameters and their correlation coefficients."""
    import argparse

    parser = argparse.ArgumentParser(
        description="calculate errors of a given set of parameters and their correlation coefficients"
    )
    parser.add_argument(
        "--params", default="final_params.json", dest="params_file"
    )
    results = parser.parse_args()

    cal_errors(params_file=results.params_file)


def cal_errors(params_file):
    config = ConfigLoader("config.yml")
    config.set_params(params_file)
    fcn = config.get_fcn()

    # current params
    params = config.get_params()

    # obtain the Hesse errors. the error matrix is cached in config.inv_he
    errors = config.get_params_error()
    np.save("error_matrix.npy", config.inv_he)

    print("\n########## fit parameters:")
    for key, value in config.get_params().items():
        print(key, error_print(value, errors.get(key, None)))

    print("\n########## error matrix:")
    print("Matrix index:\n", fcn.model.Amp.vm.trainable_vars)
    # obtain the correlation matrix using the inverse Hessian
    print("Error matrix:\n", config.inv_he)

    print("Correlation matrix:\n", corr_coef_matrix(config.inv_he))


if __name__ == "__main__":
    main()
