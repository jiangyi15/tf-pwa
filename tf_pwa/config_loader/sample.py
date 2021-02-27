from tf_pwa.amp.core import get_particle_model_name
from tf_pwa.cal_angle import cal_angle_from_momentum
from tf_pwa.data import data_mask, data_merge, data_shape
from tf_pwa.phasespace import generate_phsp as generate_phsp_o
from tf_pwa.tensorflow_wrapper import tf

from .config_loader import ConfigLoader


@ConfigLoader.register_function()
def sampling(config, N=1000, force=True):
    decay_group = config.get_decay()
    amp = config.get_amplitude()

    def gen(M):
        pi = generate_phsp(config, M)
        return cal_angle_from_momentum(pi, decay_group)

    all_data = []
    n_gen = 0
    while N > n_gen:
        data = single_sampling(gen, amp, 5 * N)
        n_gen += data_shape(data)
        all_data.append(data)

    ret = data_merge(*all_data)

    if force:
        cut = tf.range(data_shape(ret)) < N
        ret = data_mask(ret, cut)

    return ret


def single_sampling(phsp, amp, N):
    data = phsp(N)
    weight = amp(data)
    rnd = tf.random.uniform(weight.shape, dtype=weight.dtype)
    cut = rnd * tf.reduce_max(weight) * 1.1 < weight
    data = data_mask(data, cut)
    return data


@ConfigLoader.register_function()
def generate_phsp(config, N=1000):
    decay_group = config.get_decay()

    for i in decay_group:
        for j in i:
            if get_particle_model_name(j.core) == "one":
                print(j)

    m0 = decay_group.top.get_mass()
    mi = [i.get_mass() for i in decay_group.outs]
    if any(i is None for i in [m0] + mi):
        raise ValueError("mass required to generate phase space")
    pi = generate_phsp_o(float(m0), [float(i) for i in mi], N=N)
    return {k: v for k, v in zip(decay_group.outs, pi)}
