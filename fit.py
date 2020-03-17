#!/usr/bin/env python3
from tf_pwa.config_loader import ConfigLoader
from pprint import pprint
from tf_pwa.utils import error_print

def fit(init_params=None,):

    config = ConfigLoader("config.yml")
    try:
        config.set_params("init_params.json")
        print("using init_params.json")
    except Exception as e:
        print("using RANDOM parameters")

    data, phsp, bg = config.get_all_data()
    
    fit_result = config.fit(data, phsp, bg=bg, batch=65000)
    
    pprint(fit_result.params)
    fit_result.save_as("final_params.json")
    config.plot_partial_wave({}, data, phsp, bg=bg)
    fit_error = config.get_params_error(fit_result, data, phsp, bg=bg, batch=13000)
    fit_result.set_error(fit_error)
    pprint(fit_error)
    
    print("\n########## fit results:")
    for k, v in config.get_params().items():
        print(k, error_print(v, fit_error.get(k, None)))
    
    fit_frac, err_frac = config.cal_fitfractions({}, phsp)
    print("########## fit fractions")
    for i in fit_frac:
        print(i, error_print(fit_frac[i], err_frac.get(i, None)))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="simple fit scripts")
    parser.add_argument("--no-GPU", action="store_false", default=True, dest="has_gpu")
    results = parser.parse_args()
    if results.has_gpu:
        with tf.device("/device:GPU:0"):
            fit()
    else:
        with tf.device("/device:CPU:0"):
            fit()


if __name__ == "__main__":
    main()
