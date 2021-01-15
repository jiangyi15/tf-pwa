import os.path
import sys

import yaml

# import tf_pwa
from tf_pwa.config_loader import ConfigLoader

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

    plot(params_file=results.params_file)


def plot(params_file):
    config = ConfigLoader("config.yml")
    with open(params_file) as f:
        params = yaml.safe_load(f)
    params = params["value"]
    config.plot_partial_wave(
        params, plot_pull=True, prefix="fig/", save_pdf=True, save_root=True
    )


if __name__ == "__main__":
    main()
