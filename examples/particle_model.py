"""
Examples for particle model
---------------------------

decay system is model as

DecayGroup
    DecayChain

        Decay

        Particle

"""

import matplotlib.pyplot as plt

from tf_pwa.amp import Decay, DecayChain, DecayGroup, Particle
from tf_pwa.vis import plot_decay_struct

# %%
# We can easy create some instance of Particle
# and then combine them as Decay

a = Particle("A")
b = Particle("B")
c = Particle("C")
d = Particle("D")

r = Particle("R")

dec1 = Decay(a, [r, b])
dec2 = Decay(r, [c, d])

# %%
#
# DecayChain is a list of Decays.

decay_chain = DecayChain([dec1, dec2])
decay_chain

# %%
# We can plot it using matplotlib.

plot_decay_struct(decay_chain)
plt.show()

# %%
# DecayGroup is a list of DecayChain with the same initial and final states

decay_group = DecayGroup([decay_chain])
decay_group

# %%
# We can build a simple function to infer the charge from final states.


def charge_infer(dec, charge_map):
    # use particle equal condition
    cached_charge = {Particle(k): v for k, v in charge_map.items()}
    # loop for all decays in decay chain
    for i, dec_i in dec.depth_first(False):
        # all out particles has charge
        assert all(i in cached_charge for i in dec_i.outs)
        # the charge or core particle is the sum of
        cached_charge[dec_i.core] = sum(cached_charge[i] for i in dec_i.outs)
    return cached_charge


charges = {
    "B": -1,
    "C": 0,
    "D": 1,
}

charge_infer(decay_chain, charges)

# %%
# See more in `~tf_pwa.cal_angle.cal_chain_boost`.
