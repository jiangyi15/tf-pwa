
=================
FAQ
=================

1. Precission Loss
^^^^^^^^^^^^^^^^^^

  ::

      message: Desired error not nescessarily achieved due to precision loss.

1.1 check the jac value,
------------------------

1.1.1 if all absulute value is small. it is acceptable beacuse of the precision.

1.1.2 if some absulte value is large. It is the bad parameters or problem in models.


2. NaN value in fit
^^^^^^^^^^^^^^^^^^^

  ::

      message: NaN result encountered.

2.1 Check the data.
-------------------

There a script (scripts/check_nan.py) to check it.

2.1.1 No stange value in data, (nan, infs ...).

2.1.2 The data order should be :math:`E, p_x, p_y,p_z`, :math:`E` is the first.

2.1.3 The mass should be valid, :math:`E^2 - p_x^2 - p_y^2 - p_z^2 > 0`, and for any combinations of final particles, mab > ma  + mb.

2.1.4 Avoid 0 in weights.


2.2 check the model.
--------------------

2.1.1 The resonaces mass should be valid, for example in the mass range (m1+m2, m0-m3), out of the threshold required special options.

3. NaN value when getting params error.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  ::

      numpy.linalg.LinAlgError: Array must not contain infs or NaN.

3.1 similar as 2.2. Bad fit parameters: too wide width or two narrow width, reach the boundary and so on.

3.2 Bad gradients. No gradients or the gradients is not corret for fit paramters.
