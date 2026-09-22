
=================
FAQ
=================

1. Precision Loss
^^^^^^^^^^^^^^^^^^

  ::

      message: Desired error not nescessarily achieved due to precision loss.

Check the jac value,

1.1 If all absulute values are small. it is acceptable because of the precision.

1.2 If some absolute values are large. It is due to the bad parameters or some problem in the models.

1.3 Avoid negative weights

2. NaN value in fit
^^^^^^^^^^^^^^^^^^^

  ::

      message: NaN result encountered.

2.1 Check the data.
-------------------

There is a script (scripts/check_nan.py) to check it.

2.1.1 No strange value in data, (nan, infs ...).

2.1.2 The data order should be :math:`E, p_x, p_y,p_z`, :math:`E` is the first.

2.1.3 The mass should be valid, :math:`E^2 - p_x^2 - p_y^2 - p_z^2 > 0`, and for any combination of final particles, mab > ma  + mb.

2.1.4 Avoid including 0 in the weights.


2.2 Check the model.
--------------------

2.2.1 The resonances mass should be valid, for example in the mass range (m1+m2, m0-m3), out of the threshold required for special options.

3. NaN value when getting params error.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  ::

      numpy.linalg.LinAlgError: Arrays must not contain infs or NaN.

3.1 Similar as sec 2.2.

3.2 Bad fit parameters: width too narrow or wide, reach the boundary and so on.

3.3 Bad gradients. No gradients or the gradients are not correct for fit paramters.

4. Singular Matrix when getting params error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  ::

      numpy.linalg.LinAlgError: Singular matrix

4.1 Free parameters are not used in the model.

4.2 Used numpy for calculation of variable. The calculations have to be done in get_amp with TensorFlow.

  .. code::

    ...
      def init_params(self):
         self.a = self.add_var("a")
      def get_amp(self, data, *args, **kwargs):
         # avoid using numpy for variable as
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

5.1.3 Try to use a small data sample, or simpler cases (less final particles).

5.1.4 Some special models require large memory (such as an interpolation model), try another model.

5.1.5 Compile the NLL and its gradient with :code:`tf_function_nll`

  .. code::
     yaml

     # config.yml
     data:
        tf_function_nll: True

  This compiles the whole negative log-likelihood and its gradient into
  :code:`tf.function` while keeping the batch loop eager, so it also works
  with :code:`lazy_call`. It can speed up the scipy BFGS fit by a few times.

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

The decay chain should be complete. All the item in decay chain should decay from initial to final state.


6.3 Decay chain 2
-----------------

  ::

      RuntimeError: not decay chain available, check your config.yml

6.3.1 Similar as sec 6.2.

6.3.2 Check the information in *remove decay chain*, see the reason why those decays are not available.

*ls not available* means no possible LS combination allowed. Check the spin and parity. If parity is allowed to be violated, add :code:`p_break: True` to decay.
