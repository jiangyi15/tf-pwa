import os.path
import sys

import numpy as np

from tf_pwa.applications import cal_hesse_error, corr_coef_matrix

# import tf_pwa
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.utils import error_print

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir + "/..")


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
    fcn.model.Amp.vm.rp2xy_all()  # we can use this to transform all complex parameters to xy-coordinates, since the Hesse errors of xy are more statistically reliable
    params = config.get_params()
    errors, config.inv_he = cal_hesse_error(
        fcn, params, check_posi_def=True, save_npy=True
    )  # obtain the Hesse errors and the error matrix (inverse Hessian)
    print("\n########## fit parameters in XY-coordinates:")
    errors = dict(zip(fcn.model.Amp.vm.trainable_vars, errors))
    for key, value in config.get_params().items():
        print(key, error_print(value, errors.get(key, None)))

    print("\n########## correlation matrix:")
    print("Matrix index:\n", fcn.model.Amp.vm.trainable_vars)
    print(
        "Correlation Coefficients:\n", corr_coef_matrix(config.inv_he)
    )  # obtain the correlation matrix using the inverse Hessian


if __name__ == "__main__":
    main()
