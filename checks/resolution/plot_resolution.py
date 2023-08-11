import sys

from tf_pwa.config_loader import ConfigLoader


def main():
    config = ConfigLoader("config.yml")
    param = "final_params"
    if len(sys.argv) > 1:
        param = sys.argv[1]
    config.set_params(param + ".json")
    config.plot_partial_wave(prefix="figure/" + param + "_", plot_pull=True)


if __name__ == "__main__":
    main()
