import sys
import os.path
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir + '/..')
import numpy as np
#import tf_pwa
from tf_pwa.config_loader import ConfigLoader


def fit(final_params_file):
    config = ConfigLoader("ex_config.yml")
    #config.set_params("ex_gen_params.json")
    fit_result = config.fit()
    errors = config.get_params_error(fit_result)
    fit_result.set_error(errors)
    fit_result.save_as(final_params_file)
    config.plot_partial_wave(fit_result, plot_pull=True, prefix="fig/")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="complete fit")
    parser.add_argument("--final_params", default="final_params.json", dest="final_params")
    results = parser.parse_args()
    fit(final_params_file=results.final_params)
    
if __name__ == "__main__":
    main()