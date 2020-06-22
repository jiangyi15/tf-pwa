
------------
Custom Model
------------


TF-PWA support custom model of ``Particle``, just implement the ``get_amp`` method for a class inherited from Particle as:

.. code-block:: python

   from tf_pwa.amp import register_particle, Particle
   
   @register_particle("MyModel")
   class MyParticle(Particle):
       def get_amp(self, *args, **kwargs):
           print(args, kwargs)
           return 1.0

Then it can be used in ``config.yml`` (or ``Resonances.yml``) as ``model: MyModel``. 
We can get the data used for amplitude, and add some calculations such as Breit-Wigner.


.. code-block:: python

   from tf_pwa.amp import register_particle, Particle
   import tensorflow as tf
   
   @register_particle("BW")
   class SimpleBW(Particle):
       def get_amp(self, *args, **kwargs):
           """
           Breit-Wigner formula
           """
           m = args[0]["m"]
           m0 = self.get_mass()
           g0 = self.get_width()
           delta_m = m0*m0 - m * m
           one = tf.ones_like(m)
           ret = 1/tf.complex(delta_m, m0 * g0 * one)
           return ret

Note, we used ``one`` to make sure the shape to be same.

We can also add parameters in the Model ``init_params`` using ``self.add_var(...)``.

.. code-block:: python

   @register_particle("Line")
   class LineModel(Particle):
       def init_params(self):
           super(LineModel, self).init_params()
           self.a = self.add_var("a")
       def get_amp(self, data, *args, **kwargs):
           """ model as a * m + 1 """
           m = data["m"]
           zeros = tf.zeros_like(m)
           return tf.complex(self.a() * m + 1, zeros)

Then a parameters ``{particle_name}_a`` will appear in the parameters, we use ``self.a()`` to get the value in ``get_amp``. 
Note, the type of return value should be ``tf.complex``. All builtin model is located in ``tf_pwa/amp.py``.

