Resolution in Partial Wave Analysis
-----------------------------------

Resolution is the effect of detector. To Consider the resolution properly, We need to take a general look about the detector process. We can divide the process detector into two part.
The first part is acceptance, with the probability for truth value :math:`x` as :math:`\epsilon_{T} (x)`.
The second part is resolution, it means the measurement value :math:`y` will be a random number base on truth value :math:`x`. It is a conditional probability as :math:`R_{T}(y|x)`. The conditional probability is normalized as :math:`\int R_{T}(y|x) \mathrm{d} y = 1`.
So, the total effect of detector is transition function

.. math::
    T(x,y) = R_{T}(y|x)\epsilon_{T} (x).

When we have a distribution of truth value with probability :math:`p(x)`, then we can get the distribution of measurement value with probability

.. math::
    p'(y)= \int p(x) T(x,y) \mathrm{d} x.

Using the *Bayes Rule*, we can rewrite :math:`T(x,y)` as

.. math::
    T(x,y) = R(x|y) \epsilon_{R}(y),

where

.. math::
    \epsilon_{R}(y) = \int T(x,y) \mathrm{d} x, \ R(x|y) = \frac{T(x,y)}{\epsilon_{R}(y)}.

:math:`R(x|y)` is the posterior probability, that means the probability of a certain :math:`y` is from :math:`x`.
:math:`\epsilon_{R}(y)` is the projection of :math:`y` for :math:`T(x,y)`, and is also the normalize factor for :math:`S(x|y)`.

Then, the probability :math:`p'(y)` can be rewritten as

.. math::
    p'(y) =  \epsilon_{R}(y) \int p(x) S(x|y) \mathrm{d} x.

To consider the resolution, we need to determine :math:`R(x|y)`. Generally, we use simulation to determine :math:`R(x|y)`. When :math:`p(x)=1` is a flat distribution, then the joint distribution of :math:`x` and :math:`y` has the probability density :math:`T(x,y)`. We can build a model for this distribution. To get :math:`R(x|y)`, we only need to do a normalization for :math:`T(x,y)`.

In PWA, we usually use the MC to do the normalization for signal probability density. We need to calculate the integration of :math:`p'(y)` as

.. math::
   \int p'(y) \mathrm{d} y = \int p(x) \epsilon_{T} (x) \int R_{T}(y|x) \mathrm{d} y \mathrm{d} x = \int p(x) \epsilon_{T} (x) \mathrm{d} x.

The final likelihood with considering resolution is

.. math::
    - \ln L = -\sum \ln \frac{p'(y)}{\int p'(y) dy} = -\sum \ln \frac{\int p(x) R(x|y) \mathrm{d} x}{ \int p(x) \epsilon_{T} (x) \mathrm{d} x } - \sum \ln \epsilon_{R}(y).

The last part is a constant, we can ignore it in fit. In the numerical form, it can be written as

.. math::
    - \ln L = -\sum \ln \frac{1}{M}\sum_{x \sim R(x|y)} p(x) + N \ln \sum_{x \sim \epsilon_{T}(x)} p(x).

For the second part, which we already have MC sample with :math:`x \sim \epsilon_{T}(x)`, we can use MC sample to do the sum directly.
For the first part, we can generate some :math:`x` (:math:`M` times) for every :math:`y` (:math:`N` events). Using the generated samples (:math:`MN` events), we can calculate though the summation.

In addition we can insert some importance information for the summation as

.. math::
    \int p(x) R(x|y) \mathrm{d} x \approx \frac{1}{\sum w_i} \sum_{x\sim \frac{R(x|y)}{w_i(x)}} w_i p(x).

We need to keep the normalization. For example, we can use Gauss-Hermite quadrature.

In a simple situation, we only use mass for the variable for resolution function.
We can build the datasets by replacing the mass by random number based on the resolution function,
keeping the same for other variables and using some constrains.

Once we get such datasets, we can use the likelihood method to fit the dataset with resolution.
There is an example in `checks <https://github.com/jiangyi15/tf-pwa/tree/dev/checks/resolution>`_.
