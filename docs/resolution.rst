Resolution in Partial Wave Analysis
-----------------------------------

[WIP]

To consider resolution in partial wave analysis, the probability density function will change into

.. math::
    P(x) = \frac{ \int |A(x-y)|^2 R(y_i) d y }{\int |A(x)|^2 d \Phi}.

What we can do is using summation to replace integration as 

.. math::
    P(x) = \frac{ \sum_i \alpha_i |A(x-y_i)|^2 }{\int |A(x)|^2 d \Phi}.

:math:`\alpha_i` is the value of resolution function(:math:`R(y_i)`),

So, what we need is the datasets of :math:`{x-y_i}`. There is not standard way to create such datasets.
One way to do that is using reconstruction randomly for our data.

In a simple satuation, we using only mass for the resultion function variable. 
we can build the datasets by replacing the mass by random number based on the resultion function, 
keeping some variable and using some constrains.

Once we get such datasets, we can use the likelihood method to fit the dataset with resolution.

