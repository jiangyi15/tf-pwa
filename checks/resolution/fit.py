from pprint import pprint

from tf_pwa.config_loader import ConfigLoader


def main():
    config = ConfigLoader("config.yml")
    fit_result = config.fit(batch=63000)
    pprint(fit_result.params)
    fit_error = config.get_params_error(fit_result, batch=12000)
    fit_result.set_error(fit_error)
    fit_result.save_as("final_params.json")


if __name__ == "__main__":
    main()
