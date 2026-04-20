"""
sWeights implementation.

sWeights are used to statistically separate signal and background contributions
in a data sample when the distributions overlap.
"""

import numpy as np


def cal_sweights(pdfs, N=None):
    """
    Calculate sWeights for a set of PDF components.

    Given normalized PDFs for each component (signal, background, etc.),
    computes weights such that the weighted sum of events gives the yield
    for each component.

    Parameters
    ----------
    pdfs : list of array-like
        List of PDF values evaluated at each data point.
        Each element should be an array of shape (n_events,).

        Two calling conventions are supported:

        1. Normalized PDFs with N parameter:

           .. code-block:: python

               >>> import numpy as np
               >>> from scipy.stats import norm
               >>> np.random.seed(42)
               >>> x_sig = np.random.normal(5.0, 0.5, 50)
               >>> x_bkg = np.random.uniform(0, 10, 100)
               >>> x = np.concatenate([x_sig, x_bkg])
               >>> f_sig = norm.pdf(x, 5.0, 0.5)
               >>> f_bkg = np.ones_like(x) / 10.0
               >>> sw = cal_sweights([f_sig, f_bkg], N=[50, 100])
               >>> bool(np.isclose(np.sum(sw[0]), 50))
               True

        2. Unnormalized PDFs without N parameter:

           .. code-block:: python

               >>> sw2 = cal_sweights([f_sig * 50, f_bkg * 100])
               >>> bool(np.allclose(np.sum(sw2[0]), 50, rtol=0.1))
               True

    N : array-like, optional
        Expected yields for each component. If None, yields are estimated
        from the PDFs assuming the PDFs are already scaled by yields.

    Returns
    -------
    sweights : ndarray
        Array of shape (n_components, n_events) containing sWeights for each
        component at each event.

    Raises
    ------
    LinAlgError
        If the covariance matrix is singular and cannot be inverted.

    Notes
    -----
    The sWeights are computed using Cowan's formula:

    .. math::

        p(x) = \\sum_i N_i f_i(x)

    .. math::

        V_{ij}^{-1} = \\sum_k \\frac{f_i(x_k) f_j(x_k)}{p(x_k)^2}

    .. math::

        w_i(x_k) = \\frac{\\sum_j V_{ij} f_j(x_k)}{p(x_k)}

    where:

    - :math:`N_i` is the yield for component i
    - :math:`f_i(x)` is the normalized PDF for component i
    - :math:`p(x)` is the total PDF
    - :math:`V` is the covariance matrix of the yields

    Reference: `sPlot paper <https://doi.org/10.1016/j.nima.2005.08.106>`_
    """
    pdfs = np.stack(pdfs, axis=0)

    if N is None:
        # Estimate yields from unnormalized PDFs
        p = np.sum(pdfs, axis=0)
        # N = int c_i f_i(x) dx
        #   = int c_i f_i(x)/p(x) * p(x) dx
        #   = E [ c_i f_i(x) / p(x) ]
        ratio = np.sum(pdfs / p, axis=-1)
        N = p.shape[0] * ratio / np.sum(ratio)
        pdfs = pdfs / N[:, None]
    else:
        N = np.array(N)
        p = np.sum(N[:, None] * pdfs, axis=0)

    Vij_inv = np.sum(pdfs[:, None, :] * pdfs[None, :, :] / p**2, axis=-1)
    Vij = np.linalg.inv(Vij_inv)
    sw = np.matmul(Vij, pdfs) / p

    return sw
