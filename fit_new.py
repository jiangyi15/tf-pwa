#!/usr/bin/env python3
from tf_pwa.config_loader import ConfigLoader


def fit():
    config = ConfigLoader("config.yml.sample")
    data, phsp, bg = config.get_all_data()
    fit_result = config.fit(data, phsp, bg=bg, batch=65000)
    fit_error = config.cal_error(fit_result, data, phsp, bg=bg, batch=13000)
    fit_result.saveas("final_params.json")

if __name__ == "__main__":
    fit()
