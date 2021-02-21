"""
Basic Amplitude Calculations.
A partial wave analysis process has following structure:

DecayGroup: addition (+)
    DecayChain: multiplication (x)
        Decay, Particle(Propagator)

"""
from .base import *

# pylint: disable=unused-wildcard-import,unused-import
from .core import *
from .flatte import ParticleFlatte
