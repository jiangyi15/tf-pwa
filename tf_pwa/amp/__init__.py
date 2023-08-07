"""
Basic Amplitude Calculations.
A partial wave analysis process has following structure:

DecayGroup: addition (+)
    DecayChain: multiplication (x)
        Decay, Particle(Propagator)

"""

# pylint: disable=unused-wildcard-import,unused-import
from .amp import AmplitudeModel, create_amplitude
from .base import *
from .core import *
from .flatte import ParticleFlatte
from .Kmatrix import KmatrixSingleChannelParticle
from .kmatrix_simple import KmatrixSimple
from .preprocess import create_preprocessor
from .split_ls import ParticleBWRLS
