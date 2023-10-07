
=================
FAQ
=================

1. Precission Loss
^^^^^^^^^^^^^^^^^^

  ::

      message: Desired error not nescessarily achieved due to precision loss.

Check the jac value,

1.1 If all absulute value is small. it is acceptable beacuse of the precision.

1.2 If some absulte value is large. It is the bad parameters or problem in models.

1.3 Avoid negative weights

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


2.2 Check the model.
--------------------

2.2.1 The resonaces mass should be valid, for example in the mass range (m1+m2, m0-m3), out of the threshold required special options.

3. NaN value when getting params error.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  ::

      numpy.linalg.LinAlgError: Array must not contain infs or NaN.

3.1 Similar as sec 2.2.

3.2 Bad fit parameters: too wide width or two narrow width, reach the boundary and so on.

3.3 Bad gradients. No gradients or the gradients is not corret for fit paramters.

4. Singular Matrix when getting params error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  ::

      numpy.linalg.LinAlgError: Singular matrix

4.1 Free paramters are not used in model.

4.2 Used numpy for calculation of variable. The calculation have to be done in get_amp with TensorFlow.

  .. code::

    ...
      def init_params(self):
         self.a = self.add_var("a")
      def get_amp(self, data, *args, **kwargs):
         # avoid use numpy for varible as
         a = np.sin(self.a())
         # use tensorflow instead
         a = tf.sin(self.a())

5. Out of memory (OOM)
^^^^^^^^^^^^^^^^^^^^^^

5.1 GPU
-------------------

  ::

      tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor with shape ... device:GPU:0 by allocator GPU_0_bfc [Op:...]

5.1.1 Reduce batch size at :code:`config.fit(batch=65000)` and `config.get_params_error(batch=13000)` in fit.py.

5.1.2 Use option for large data size, such as lazy call

  .. code::
     yaml

     # config.yml
     data:
        lazy_call: True

5.1.3 Try to use small data sample, or simple cases (less final particles).

5.1.4 Some speical model required large memory (such as interpolation model), try other model.

5.2 CPU
-------------------

  ::

      killed

5.2.1 Try to allocate more memory. There should be some options for job.

5.2.2 Similar as sec 5.1

6. Bad config.yml
^^^^^^^^^^^^^^^^^

6.1 yaml parse error
--------------------

  ::

      yaml.parser.ParserError: while parsing ..

Check the yaml file (see https://yaml.org): the indent, speical chars :code:`,:}]`, unicode and so on.

6.2 Decay chain
---------------

  ::

      AssertionError: not only one top particle

The decay chain should be complete. All the item in decay should decay from initial to finals.


6.3 Decay chain 2
-----------------

  ::

      RuntimeError: not decay chain aviable, check you config.yml

6.3.1 Similar as sec 6.2.

6.3.2 Check the information in *remove decay chain*, see the reson why those decays are not aviable.

*ls not aviable* means no possible LS combination allowed. Check the spin and parity. If allow parity voilate, add :code:`p_break: True` to decay.
