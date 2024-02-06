from tf_pwa.cov_ten_ir import *


def test_SCombLS():
    SCombLS(1, 3 / 2, 3 / 2, 0)
    SCombLS(1, 3 / 2, 3 / 2, 1)
    SCombLS(1, 3 / 2, 3 / 2, 2)
    SCombLS(1, 3 / 2, 3 / 2, 3)
