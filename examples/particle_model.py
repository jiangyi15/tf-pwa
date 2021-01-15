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
# DecayGroup is a list of DEcayChain with the same initial and final states

decay_group = DecayGroup([decay_chain])
decay_group

# %%
# We can build a simple function to infer the charge from final states.


def charge_infer(dec, charge_map):
    # use particle equal condition
    cached_charge = {Particle(k): v for k, v in charge_map.items()}
    # loop for all decays in decay chain
    dec_set = list(dec)
    while dec_set:
        # get the first decay
        dec_i = dec_set.pop(0)
        # if all out particles has charge
        if all(i in cached_charge for i in dec_i.outs):
            # the charge or core particle is the sum or
            cached_charge[dec_i.core] = sum(
                cached_charge[i] for i in dec_i.outs
            )
            continue
        # wait for another loop
        dec_set.append(dec_i)
    return cached_charge


charges = {
    "B": -1,
    "C": 0,
    "D": 1,
}

charge_infer(decay_chain, charges)

# %%
# See more in `~tf_pwa.cal_angle.cal_chain_boost`.
