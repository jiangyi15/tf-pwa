"""
Particle and amplitude
---------------------------

    Amplitude = DecayGroup + Variable

"""

# %%
# We will use following parameters for a toy model
from tf_pwa.amp import DecayChain, DecayGroup, get_decay, get_particle

resonances = {
    "R0": {"J": 0, "P": 1, "mass": 1.0, "width": 0.07},
    "R1": {"J": 0, "P": 1, "mass": 1.0, "width": 0.07},
    "R2": {"J": 1, "P": -1, "mass": 1.225, "width": 0.08},
}

a, b, c, d = [get_particle(i, J=0, P=-1) for i in "ABCD"]
r1, r2, r3 = [get_particle(i, **resonances[i]) for i in resonances.keys()]


decay_group = DecayGroup(
    [
        DecayChain([get_decay(a, [r1, c]), get_decay(r1, [b, d])]),
        DecayChain([get_decay(a, [r2, b]), get_decay(r2, [c, d])]),
        DecayChain([get_decay(a, [r3, b]), get_decay(r3, [c, d])]),
    ]
)

# %%
# The above parts can be represented as config.yml used by ConfigLoader.
#
# We can get AmplitudeModel form decay_group and a optional Variables Managerager.
# It has parameters, so we can get and set parameters for the amplitude model
from tf_pwa.amp import AmplitudeModel
from tf_pwa.variable import VarsManager

vm = VarsManager()
amp = AmplitudeModel(decay_group, vm=vm)

print(amp.get_params())
amp.set_params(
    {
        "A->R0.CR0->B.D_total_0r": 1.0,
        "A->R1.BR1->C.D_total_0r": 1.0,
        "A->R2.BR2->C.D_total_0r": 7.0,
    }
)

# %%
# For the calculation, we generate some phase space data.

from tf_pwa.phasespace import PhaseSpaceGenerator

m_A, m_B, m_C, m_D = 1.8, 0.18, 0.18, 0.18
p1, p2, p3 = PhaseSpaceGenerator(m_A, [m_B, m_C, m_D]).generate(100000)

# %%
# and the calculate helicity angle from the data

from tf_pwa.cal_angle import cal_angle_from_momentum

data = cal_angle_from_momentum({b: p1, c: p2, d: p3}, decay_group)

# %%
# we can index mass from data as

from tf_pwa.data import data_index

m_bd = data_index(data, ("particle", "(B, D)", "m"))
# m_bc = data_index(data, ("particle", "(B, C)", "m"))
m_cd = data_index(data, ("particle", "(C, D)", "m"))

# %%
# .. note::
#     If DecayGroup do not include resonant of (B, C), the data will not include its mass too.
#     We can use different DecayGroup for cal_angle and AmplitudeModel
#     when they have the same initial and final particle.
#
# The amplitde square is calculated by amplitude model simply as

amp_s2 = amp(data)

# %%
# Now by using matplotlib we can get the Dalitz plot as

import matplotlib.pyplot as plt

plt.clf()
plt.hist2d(
    m_bd.numpy() ** 2,
    m_cd.numpy() ** 2,
    weights=amp_s2.numpy(),
    bins=60,
    cmin=1,
    cmap="jet",
)
plt.colorbar()
plt.xlabel("$m^2(BD)$")
plt.ylabel("$m^2(CD)$")
plt.show()
