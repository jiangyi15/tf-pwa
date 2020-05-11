import sys
import os.path
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir + '/..')
import numpy as np
#import tf_pwa
from tf_pwa.applications import gen_data, gen_mc
from tf_pwa.config_loader import ConfigLoader


def generate_phspMC(Nmc, mc_file):
    mA = 4.6
    mB = 2.00698
    mC = 2.01028
    mD = 0.13957
    a2bcd = gen_mc(mA, [mB, mC, mD], Nmc)
    np.savetxt(mc_file, a2bcd)

def generate_toy_from_phspMC(Ndata, mc_file, data_file):
    config = ConfigLoader("ex_config.yml")
    config.set_params("ex_gen_params.json")
    amp = config.get_amplitude()
    data = gen_data(amp, Ndata=Ndata, mcfile=mc_file, genfile=data_file)
    return data

def main():
    import argparse
    parser = argparse.ArgumentParser(description="generate toy of a certain amplitude")
    parser.add_argument("--Nmc", default=2000, dest="Nmc")
    parser.add_argument("--Ndata", default=100, dest="Ndata")
    results = parser.parse_args()
    
    if not os.path.exists("ex_data"):
        os.mkdir("ex_data")
    generate_phspMC(Nmc=results.Nmc, mc_file="ex_data/ex_PHSP.dat")
    generate_toy_from_phspMC(Ndata=results.Ndata, mc_file="ex_data/ex_PHSP.dat", data_file="ex_data/ex_data.dat")

if __name__ == "__main__":
    main()

