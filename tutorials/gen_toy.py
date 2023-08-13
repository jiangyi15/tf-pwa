import os.path
import sys

import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir + "/..")

# import tf_pwa
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.phasespace import PhaseSpaceGenerator


def main():
    """Take three-body decay A->BCD for example, we generate a PhaseSpace MC sample and a toy data sample."""
    import argparse

    parser = argparse.ArgumentParser(
        description="generate toy of a certain amplitude"
    )
    parser.add_argument("--Nmc", default=2000, type=int, dest="Nmc")
    parser.add_argument("--Ndata", default=100, type=int, dest="Ndata")
    results = parser.parse_args()

    if not os.path.exists("data"):
        os.mkdir("data")

    generate_phspMC(Nmc=results.Nmc, mc_file="data/PHSP.dat")
    generate_toy_from_phspMC(Ndata=results.Ndata, data_file="data/data.dat")


def generate_phspMC(Nmc, mc_file):
    """Generate PhaseSpace MC of size Nmc and save it as txt file"""
    # We use ConfigLoader to read the information in the configuration file
    config = ConfigLoader("config.yml")
    # Set the parameters in the amplitude model
    config.set_params("gen_params.json")

    phsp = config.generate_phsp_p(Nmc)

    config.data.savetxt(mc_file, phsp)


def generate_toy_from_phspMC(Ndata, data_file):
    """Generate toy using PhaseSpace MC from mc_file"""
    # We use ConfigLoader to read the information in the configuration file
    config = ConfigLoader("config.yml")
    # Set the parameters in the amplitude model
    config.set_params("gen_params.json")

    data = config.generate_toy_p(Ndata)

    config.data.savetxt(data_file, data)
    return data


if __name__ == "__main__":
    main()
