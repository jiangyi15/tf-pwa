import warnings
from .experimental import extra_amp, extra_data

warnings.warn(
    "'expemental' is a wrong word, use `experimental` instead.",
    DeprecationWarning,
    stacklevel=2,
)
