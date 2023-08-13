Resolution in Partial Wave Analysis
-----------------------------------

Resolution is the effect of detector. To Consider the resolution properly, We need take a general look about the detector process. We can divided the proecess detector into two part.
The first part is acceptance, with the probability for truth value :math:`x` as :math:`\epsilon_{T} (x)`.
The second part is resolution, it means the measurement value :math:`y` will be a random number base on truth value :math:`x`. It the a conditional probability as :math:`R(y|x)`.
So, the total effect of detector is transition function

.. math::
    T(x,y) = R(y|x)\epsilon_{T} (x).

When we have a distribution of truth value with probability :math:`p(x)`, then we can get the distribution of measurement value with probability

.. math::
    p'(y)= \int p(x) T(x,y) dx.

Using the *Bayes Rule*, we can rewrite :math:`T(x,y)` as

.. math::
    T(x,y) = S(x|y) \epsilon_{R}(y),

where

.. math::
    \epsilon_{R}(y) = \int T(x,y) d x, \ S(x|y) = \frac{T(x,y)}{\epsilon_{R}(y)}.

:math:`S(x|y)` is the posterior probability, that means the probobility of a certain :math:`y` is from :math:`x`.
:math:`\epsilon_{R}(y)` is the projection of :math:`y` for :math:`T(x,y)`, and is also the normalisze factor for :math:`S(x|y)`.

Then, the probability :math:`p'(y)` can be rewriten as

.. math::
    p'(y) =  \epsilon_{R}(y) \int p(x) S(x|y) dx.

To consider the resolution, we need to determinate :math:`S(x|y)`. Generally, we use simulation to determinate :math:`S(x|y)`. When :math:`p(x)=1` is a flat distribution, then the joint distribution of :math:`x` and :math:`x` has the probability :math:`T(x,y)`. We can build a model for this distribution. To get :math:`S(x|y)`, we only need to do a normalization for :math:`T(x,y)`.

In PWA, we usually use the MC to do the normalization for signal probability density. We need to calulation the intergration of :math:`p'(y)` as

.. math::
   \int p'(y) dy = \int p'(x) \epsilon_{T} (x) \int R(y|x) dy dx = \int p(x) \epsilon_{T} (x) dx.

The final likilihood with considering resolution is

.. math::
    - \ln L = -\sum \ln \frac{p'(y)}{\int p'(y) dy} = -\sum \ln \frac{\int p(x) S(x|y) dx}{ \int p(x) \epsilon_{T} (x) dx } - \sum \ln \epsilon_{R}(y).

The last part is a constant, we can ignore it in fit. In the numerical form, it can be written as

.. math::
    - \ln L = -\sum \ln \frac{1}{M}\sum_{x \sim S(x|y)} p(x) + N \ln \sum_{x \sim \epsilon_{T}(x)} p(x).

For the second part, which we already have MC sample with :math:`x \sim \epsilon_{T}(x)`, we can use MC sample to do the sum directly.
For the first part, we can generate some :math:`x` (:math:`M` times) for every :math:`y` (:math:`N` events). Using the generated samples (:math:`MN` events), we can calculated though the summation.

In addtion we can insert some importance information for the summution as

.. math::
    \int p(x) S(x|y) dx \approx \frac{1}{\sum w_i} \sum_{x\sim \frac{S(x|y)}{w_i(x)}} w_i p(x).

We need to keep the normalization. For example we can use Gauss-Hermite quadrature.

In a simple satuation, we only use mass for the variable for resultion function.
We can build the datasets by replacing the mass by random number based on the resultion function,
keeping the same for other variables and using some constrains .

Once we get such datasets, we can use the likelihood method to fit the dataset with resolution.
There is an example in `checks <https://github.com/jiangyi15/tf-pwa/tree/dev/checks/resolution>`_.
