import pytest

from tf_pwa.amp import *
from tf_pwa.cal_angle import *
from tf_pwa.phasespace import PhaseSpaceGenerator


def generate_mc(num):
    a = PhaseSpaceGenerator(4.59925, [2.01026, 0.13957061, 2.00685])
    flat_mc_data = a.generate(num)
    return flat_mc_data


@pytest.mark.benchmark(group="generate_mc")
def test_generate_mc_CPU(benchmark):
    with tf.device("CPU:0"):
        benchmark(generate_mc, 1000000)


@pytest.mark.benchmark(group="generate_mc")
def test_generate_mc(benchmark):
    benchmark(generate_mc, 1000000)


def _get_decay_data(num=1):
    data = generate_mc(200000)
    a = Particle("A", J=1, P=-1, spins=(-1, 1))
    b = Particle("B", J=1, P=-1)
    c = Particle("C", J=0, P=-1)
    d = Particle("D", J=1, P=-1)
    res = Particle("res", 1, 1, mass=4.4232, width=0.025)
    dec1 = HelicityDecay(a, [res, c])
    dec2 = HelicityDecay(res, [b, d])
    if num == 1:
        decs = DecayGroup([[dec1, dec2]])
    else:
        res2 = Particle("res2", 1, 1, mass=2.4232, width=0.025)
        dec3 = HelicityDecay(a, [res2, d])
        dec4 = HelicityDecay(res2, [b, c])
        decs = DecayGroup([[dec1, dec2], [dec3, dec4]])
    p = dict(zip([d, b, c], data))
    amp = AmplitudeModel(decs)
    data = cal_angle_from_momentum(p, decs)
    args = amp.trainable_variables
    return amp, data, args


@pytest.mark.benchmark(group="mc_int")
def test_mc_int(benchmark):
    amp, data, params = _get_decay_data(1)

    def mc_int(dat):
        return tf.reduce_sum(amp(dat))

    benchmark(mc_int, data)


@pytest.mark.benchmark(group="mc_int")
def test_mc_int_CPU(benchmark):
    amp, data, params = _get_decay_data(1)

    def mc_int(dat):
        return tf.reduce_sum(amp(dat))

    with tf.device("CPU:0"):
        benchmark(mc_int, data)


@pytest.mark.benchmark(group="mc_int")
def test_mc_int2(benchmark):
    amp, data, params = _get_decay_data(2)

    def mc_int(dat):
        return tf.reduce_sum(amp(dat))

    benchmark(mc_int, data)


@pytest.mark.benchmark(group="mc_int")
def test_mc_int2_CPU(benchmark):
    amp, data, params = _get_decay_data(2)

    def mc_int(dat):
        return tf.reduce_sum(amp(dat))

    with tf.device("CPU:0"):
        benchmark(mc_int, data)


@pytest.mark.benchmark(group="mc_int")
def test_mc_int_grad(benchmark):
    amp, data, params = _get_decay_data(1)

    def mc_int(dat):
        with tf.GradientTape() as tape:
            int_mc = tf.reduce_sum(amp(dat))
        args = tape.gradient(int_mc, params)
        return int_mc, args

    benchmark(mc_int, data)


@pytest.mark.benchmark(group="mc_int")
def test_mc_int_grad_CPU(benchmark):
    amp, data, params = _get_decay_data(1)

    def mc_int(dat):
        with tf.GradientTape() as tape:
            int_mc = tf.reduce_sum(amp(dat))
        args = tape.gradient(int_mc, params)
        return int_mc, args

    with tf.device("CPU:0"):
        benchmark(mc_int, data)
