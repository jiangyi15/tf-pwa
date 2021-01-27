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
