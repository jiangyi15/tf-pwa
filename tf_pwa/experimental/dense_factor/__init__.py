"""Dense factorized amplitude model plugin.

Importing this module attaches the dense term interface to the original
tf-pwa classes, so the upstream source files stay untouched::

    import tf_pwa.experimental.dense_factor

The amplitude is written as

.. math::

    A_{ij} = \\sum_u C_u \\frac{F_{u,ij}(x)}
             {\\prod_l \\sum_{m,n} M1_{u,l,m} g_m M2_{m,n} B_{l,n}(x)}

The decay group builds the data-only ``F`` and ``B`` tensors, together
with the structural ``M2``, in ``get_dense_structure`` on first use.  The amplitude
model builds its contraction layout ``M1`` in ``init_params``.  All
coefficients are rebuilt on every call.

This module must be imported before the amplitude is created, because the
``dense_factor`` preprocessor and amplitude model are registered on import.

The implementation is split into sibling modules, all imported here so the
package keeps one import surface:

``backend``      the contraction-backend registry (a leaf, imported first)
``compress``     the compression strategies and the record store (a leaf)
``decays``       the decay/chain/group basis mixins (attached on import)
``particles``    the particle term mixins (attached on import)
``interface``    the ``terms``/``basis`` decorators (a leaf)
``preprocess``   both preprocessors
``amp``          both amplitude models
"""

import contextlib
import itertools
import warnings
import weakref

import numpy as np
import tensorflow as tf

# The names below are not used here: they are kept so the module namespace is
# unchanged by the split (``dense_factor.<name>`` still resolves).  Do not
# "clean" them; the pylint exemption is for this block.
import tf_pwa.amp.interpolation  # noqa: F401  (register spline models)
from tf_pwa.amp.amp import BaseAmplitudeModel, register_amp_model
from tf_pwa.amp.preprocess import (
    BasePreProcessor,
    CachedAnglePreProcessor,
    register_preprocessor,
)
from tf_pwa.cal_angle import cal_angle_from_momentum
from tf_pwa.config import get_config, regist_config
from tf_pwa.data import data_merge, split_generator
from tf_pwa.phasespace import PhaseSpaceGenerator

# The submodules are imported in dependency order: ``backend`` is the leaf the
# others build their tables on, and the two attached mixin modules register
# their decorators on import.  Every name is re-exported, so
# ``from tf_pwa.experimental.dense_factor import <name>`` keeps working.
# isort: off
from .backend import (
    BACKENDS,
    Backend,
    pick_backend,
)

# The compression strategies and the record store live in a sibling module.
# It is a leaf (it imports nothing from this package).

from .compress import (  # noqa: F401
    _COMPRESSION_DEFAULTS,
    _COMPRESSION_SOURCES,
    _COMPRESSIONS,
    _MERGE_KEY_COUNTER,
    _OP_FAILURES,
    _TERMS,
    _axis_groups,
    _BasisIndexCompression,
    _cast_like,
    _Compression,
    _compression_of,
    _expand_block,
    _expand_svd,
    _find_merge,
    _IndexCompression,
    _join_complex,
    _NoCompression,
    _pack_complex16,
    _pack_like,
    _project,
    _real_dtype,
    _RealIndexCompression,
    _RealSvdCompression,
    _reconstruct,
    _sparse_basis,
    _split_complex,
    _SvdCompression,
    _unpack_real,
    _unpack_stored,
    _unpack_terms,
    _warn_op_failure,
    basis_key,
    get_merge_record,
    MERGE_RECORDS,
    merge_record,
    set_merge_record,
)

from .decays import (  # noqa: F401
    _AmpDecayBasis,
    _DecayChainBasis,
    _DecayGroupBasis,
    _HelicityBasis,
)

# The mixins live in sibling modules so this file keeps the preprocessor and
# the model; the strategies are in ``compress.py``.  Importing them here runs
# their decorators, which is what attaches the interface on import.

from .particles import (  # noqa: F401
    _BWTerms,
    _DefaultTerms,
    _FlatteTerms,
    _GSTerms,
    _OneTerms,
    _SplineTerms,
)

from .interface import (  # noqa: F401
    _BASIS_TARGETS,
    AmpDecay,
    DecayChain,
    DecayGroup,
    HelicityDecay,
    _attach,
    _guard,
    basis,
    get_particle_model,
    rename_data_dict,
    terms,
)

# ---------------------------------------------------------------------------
# The models (the interface and the mixins above must be imported first)
# ---------------------------------------------------------------------------

from .preprocess import (  # noqa: F401
    DenseFactorMergePreProcessor,
    DenseFactorPreProcessor,
)
from .amp import (  # noqa: F401
    _dense_contract,
    DenseFactorMergeModel,
    DenseFactorModel,
)

# isort: on
