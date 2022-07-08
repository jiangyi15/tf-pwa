import os
import sys

import pytest

from .test_full import gen_toy


def test_fit(gen_toy):
    print(os.getcwd())
    sys.argv = [
        "fit.py",
        "-c",
        "tf_pwa/tests/config_toy.yml",
        "-i",
        "tf_pwa/tests/exp_params.json",
        "--no-GPU",
    ]
    exec("from fit import main; main()")


def test_fit2(gen_toy):
    print(os.getcwd())
    sys.argv = [
        "fit.py",
        "-c",
        "tf_pwa/tests/config_toy.yml",
        "-i",
        "tf_pwa/tests/exp_params.json",
        "--no-GPU",
        "--printer=normal",
    ]
    exec("from fit import main; main()")


def test_main_fit(gen_toy):
    from tf_pwa.app.fit import fit

    fit("tf_pwa/tests/config_toy.yml", "tf_pwa/tests/exp_params.json")
