"""
Examples for error propagation
------------------------------

Hear we use the same config in `particle_amplitude.py`

"""

config_str = """
decay:
    A:
       - [R1, B]
       - [R2, C]
       - [R3, D]
    R1: [C, D]
    R2: [B, D]
    R3: [B, C]

particle:
    $top:
       A: { mass: 1.86, J: 0, P: -1}
    $finals:
       B: { mass: 0.494, J: 0, P: -1}
       C: { mass: 0.139, J: 0, P: -1}
       D: { mass: 0.139, J: 0, P: -1}
    R1: [ R1_a, R1_b ]
    R1_a: { mass: 0.7, width: 0.05, J: 1, P: -1}
    R1_b: { mass: 0.5, width: 0.05, J: 0, P: +1}
    R2: { mass: 0.824, width: 0.05, J: 0, P: +1}
    R3: { mass: 0.824, width: 0.05, J: 0, P: +1}

"""

import matplotlib.pyplot as plt
import yaml

from tf_pwa.config_loader import ConfigLoader
from tf_pwa.histogram import Hist1D

config = ConfigLoader(yaml.full_load(config_str))
input_params = {
    "A->R1_a.BR1_a->C.D_total_0r": 6.0,
    "A->R1_b.BR1_b->C.D_total_0r": 1.0,
    "A->R2.CR2->B.D_total_0r": 2.0,
    "A->R3.DR3->B.C_total_0r": 1.0,
}
config.set_params(input_params)

data = config.generate_toy(1000)
phsp = config.generate_phsp(10000)


# %%
# After we calculated the parameters error, we will have an error matrix `config.inv_he` (using the inverse hessain).
# It is possible to save such matrix directly by `numpy.save` and to load it by `numpy.load`.

config.get_params_error(data=[data], phsp=[phsp])

# %%
# We can use the following method to profamance the error propagation
#
# .. math::
#    \sigma_{f} = \sqrt{\frac{\partial f}{\partial x_i} V_{ij} \frac{\partial f}{\partial x_j }}
#
# by adding some calculation here. We need to use `tensorflow` functions instead of those of `math` or `numpy`.
#

import tensorflow as tf

with config.params_trans() as pt:
    a2_r = pt["A->R2.CR2->B.D_total_0r"]
    a2_i = pt["A->R2.CR2->B.D_total_0r"]
    a2_x = a2_r * tf.cos(a2_i)

# %%
# And then we can calculate the error we needed as

print(a2_x.numpy(), pt.get_error(a2_x).numpy())

# %%
# We can also calculate some more complex examples, such as the ratio in mass range (0.75, 0.85) over full phace space.
# Even further, we can get the error of error in the meaning of error propagation.

from tf_pwa.data import data_mask

m_R2 = phsp.get_mass("(B, D)")
cut_cond = (m_R2 < 0.85) & (m_R2 > 0.75)

amp = config.get_amplitude()

with config.params_trans() as pt1:
    with config.params_trans() as pt:
        int_mc = tf.reduce_sum(amp(phsp))
        cut_phsp = data_mask(phsp, cut_cond)
        cut_int_mc = tf.reduce_sum(amp(cut_phsp))
        ratio = cut_int_mc / int_mc
    error = pt.get_error(ratio)

print(ratio.numpy(), "+/-", error.numpy())
print(error.numpy(), "+/-", pt1.get_error(error).numpy())
