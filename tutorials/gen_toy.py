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
    # the order of A->B,C,D should be same as dat_order in config.yml:data: [B, C, D]
    mA = 4.6  # masses of mother particle A and daughters BCD
    mB = 2.00698
    mC = 2.01028
    mD = 0.13957

    phsp_gen = PhaseSpaceGenerator(mA, [mB, mC, mD])
    pa, pb, pc = phsp_gen.generate(Nmc)

    # a2bcd is a [3*Nmc, 4] array, which are the momenta of BCD in the rest frame of A
    a2bcd = np.concatenate([pa, pb, pc], axis=-1)

    np.savetxt(mc_file, a2bcd.reshape((-1, 4)))


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
