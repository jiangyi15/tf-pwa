#!/usr/bin/env python3
from tf_pwa.config_loader import ConfigLoader


def fit():
    config = ConfigLoader("config.yml.sample")
    data, phsp, bg = config.get_all_data()
    config.plot_partial_wave({}, data, phsp, bg=bg)
    fit_result = config.fit(data, phsp, bg=bg, batch=65000)
    print(fit_result)
    fit_result.save_as("final_params.json")
    fit_error = config.get_params_error(fit_result, data, phsp, bg=bg, batch=13000)
    print(fit_error)


if __name__ == "__main__":
    fit()
