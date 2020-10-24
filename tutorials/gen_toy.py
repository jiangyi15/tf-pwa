import sys
import os.path

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir + "/..")
import numpy as np

# import tf_pwa
from tf_pwa.applications import gen_data, gen_mc
from tf_pwa.config_loader import ConfigLoader


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
    generate_toy_from_phspMC(
        Ndata=results.Ndata, mc_file="data/PHSP.dat", data_file="data/data.dat"
    )


def generate_phspMC(Nmc, mc_file):
    """Generate PhaseSpace MC of size Nmc and save it as txt file"""
    mA = 4.6  # masses of mother particle A and daughters BCD
    mB = 2.00698
    mC = 2.01028
    mD = 0.13957
    a2bcd = gen_mc(
        mA, [mB, mC, mD], Nmc
    )  # a2bcd is a [3*Nmc, 4] array, which are the momenta of BCD in the rest frame of A
    np.savetxt(mc_file, a2bcd)


def generate_toy_from_phspMC(Ndata, mc_file, data_file):
    """Generate toy using PhaseSpace MC from mc_file"""
    config = ConfigLoader(
        "config.yml"
    )  # We use ConfigLoader to read the information in the configuration file
    config.set_params(
        "gen_params.json"
    )  # Set the parameters in the amplitude model
    amp = config.get_amplitude()
    data = gen_data(
        amp, Ndata=Ndata, mcfile=mc_file, genfile=data_file
    )  # data is saved in data_file
    return data


if __name__ == "__main__":
    main()
