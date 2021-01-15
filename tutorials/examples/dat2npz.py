import numpy as np


def dat2npz(in_file, out_file):
    data = np.loadtxt(in_file)
    np.savez(out_file, data)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("in_file")
    parser.add_argument("out_file")
    results = parser.parse_args()
    dat2npz(results.in_file, results.out_file)


if __name__ == "__main__":
    main()
