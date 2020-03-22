import pytest
from tf_pwa.data import *
from tf_pwa.particle import BaseDecay, BaseParticle
from .common import write_temp_file


def test_index():
    data = {"s": [{"s": 1}, {"d": 2}], "p": (3, 4)}
    assert data_index(data, ("s", 1)) == {"d": 2}
    assert data_index(data, ["p", 0]) == 3
    assert data_index(data, "p") == (3, 4)
    with pytest.raises(ValueError):
        data_index(data, "t")


def test_index2():
    a = BaseParticle("A")
    b = BaseParticle("B")
    c = BaseParticle("C")
    d = BaseDecay(a, [b, c])
    data = {"particle": {a: {"p": 1}, b: {"m": 2}},
            "decay": [{d: {b: {"a": 3}, c: {"a": 4}}}, ]}
    assert data_index(data, ["particle", a, "p"]) == 1
    assert data_index(data, ["particle", "B", "m"]) == 2
    assert data_index(data, ["decay", 0, str(d), b, "a"]) == 3
    assert data_index(data, ["decay", 0, d, str(c), BaseParticle("a")]) == 4


def test_flatten_dict_data():
    a = BaseParticle("A")
    b = BaseParticle("B")
    c = BaseParticle("C")
    d = BaseDecay(a, [b, c])
    data = {"particle": {a: {"p": 1}, b: {"m": 2}},
            "decay": [{d: {b: {"a": 3}, c: {"a": 4}}}, ]}
    data_f = flatten_dict_data(data)
    assert len(data_f) == 4
    assert data_f["particle/A/p"] == 1
    assert data_f["particle/B/m"] == 2
    assert data_f["decay/0/{}/B/a".format(d)] == 3
    assert data_f["decay/0/{}/C/a".format(d)] == 4


def test_load_dat_file():
    s1 = "5.0 2.0 3.0 3.0\n"
    s2 = "4.0 2.0 4.0 2.0\n4.0 2.0 4.0 2.0\n"
    print(s1 + s2)
    with write_temp_file(s1+s2) as fname:
        dat1 = load_dat_file(fname, ["a", "b", "c"])
    with write_temp_file(s1) as fname1:
        with write_temp_file(s2) as fname2:
            dat2 = load_dat_file([fname1, fname2], ["a", "b", "c"])
    assert np.allclose(dat1["a"], dat2["a"])
    assert np.allclose(dat1["b"], dat2["b"])
    assert np.allclose(dat1["c"], dat2["c"])


def test_save_load():
    b = BaseParticle("b")
    data = {"a": np.array([1.0,2.0,3.0]),
            b: np.array([2.0, 3.0, 4.0])}
    
    fname = "test_save_load.npy"
    save_data(fname, data)
    data2 = load_data(fname)
    assert np.allclose(data["a"], data2["a"])
    assert np.allclose(data[b], data2[b])
    
