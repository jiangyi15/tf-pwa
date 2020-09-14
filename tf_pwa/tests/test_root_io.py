from tf_pwa import root_io
import numpy as np


def test_read_load():
    dic = {"a": [1.0, 2.03]}
    root_io.save_dict_to_root(dic, "test_io.root", "data")
    dic2 = root_io.load_root_data("test_io.root")
    assert np.allclose(dic2["data0"]["a"].numpy(), [1.0, 2.03])
