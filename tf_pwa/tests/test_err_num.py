import math

from tf_pwa.err_num import *


def test_add():
    a = NumberError(1.0, 0.3)
    b = NumberError(2.0, 0.4)
    c = a + b
    assert c.value == 3.0
    assert c.error == 0.5
    d = b - a
    assert d.value == 1.0
    assert d.error == 0.5
    e = -a
    assert e.value == -1.0
    assert e.error == 0.3
    f = a - 1.0
    d = a + 3.0


def test_mul():
    a = NumberError(3.0, 0.3)
    b = NumberError(2.0, 0.4)
    c = a * b
    assert c.value == 6.0
    assert c.error == math.sqrt(1.8)
    d = a / b
    assert d.value == 1.5
    assert d.error == math.sqrt(0.1125)
    e = b**3
    assert e.value == 8.0
    assert abs(e.error - math.sqrt(23.04)) < 1e-7
    f = b**a
    g = 3.0**a


def test_exp():
    a = NumberError(3.0, 0.3)
    b = NumberError(2.0, 0.4)
    c = a.exp()
    d = b.log()
    e = a.apply(math.sqrt)
    g = a.apply(math.sin, math.cos)
    print(g)


def test_cal_err():
    a = NumberError(3.0, 0.3)
    b = NumberError(2.0, 0.3)
    f = lambda x: x * 3.0 + x**2 * 2.0 + 1.0
    c = cal_err(f, a)
    g = lambda x, y: x + y + (x - y) * (x + y)
    d = cal_err(g, a, b)
    e = cal_err(g, 3.0, b)
    h = cal_err(g, a, 2.0)
    assert d.value == e.value
    assert e.value == h.value
