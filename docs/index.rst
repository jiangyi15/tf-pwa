.. TFPWA documentation master file, created by
   sphinx-quickstart on Wed Jan  1 13:02:25 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TFPWA's documentation!
=================================
TFPWA is a generic software package intended for Partial Wave Analysis (PWA).
It is developed using TensorFlow2 and the calculation is accelerated by GPU.
Users may modify the configuration file (in YML format) and write simple scripts to complete the whole analysis.
A detailed configuration file sample (with all usable parameters) can be found **here**.



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   amplitude
   custom_model
   decay_topology
   phasespace
   resolution
   api
   particle_model
   FAQ


For starters
============
Take the decay in :file:`config_sample.yml` for example, we will generate a toy, and then use it to conduct an analysis.

First, we use the class `.ConfigLoader` to load the configuration file, and thus building the amplitude expression.

.. code-block:: python

   from tf_pwa.config_loader import ConfigLoader
   config = ConfigLoader("config_sample.yml")
   amp = config.get_amplitude()

We can also use a json file to set the parameters in the amplitude formula ``config.set_params("parameters.json")``.
Otherwise, the parameters are set randomly.

Next, we can use functions in `tf_pwa.applications` to directly generate data samples.
We need to provide the masses of *A* and *B*, *C*, *D* to generate the PhaseSpace MC, and then use it to generate the toy data.

.. code-block:: python

   from tf_pwa.applications import gen_mc, gen_data
   PHSP = gen_mc(mother=4.6, daughters=[2.00698, 2.01028, 0.13957], number=100000, outfile="PHSP.dat")
   data = gen_data(amp, Ndata=5000, mcfile="PHSP.dat", genfile="data.dat")

Now that we have updated ``data.dat`` and ``PHSP.dat``, we'd better load the configuration file again,
and then fit the data.

.. code-block:: python

   config = ConfigLoader("config_sample.yml")
   fit_result = config.fit()

Fitting is the major part of an analysis, and it could also be the most time-consuming part.
For this example (the complexity of the amplitude expression matters a lot), the time for fitting is about xxx (running on xxxGPU).
Then we can step further to complete the analysis, like calculating the fit fractions.

.. code-block:: python

   errors = config.get_params_error(fit_result)
   fit_result.save_as("final_parameters.json")
   fit_frac, err_frac = config.cal_fitfractions()

We can use `.error_print` in `tf_pwa.utils` to print the fitting parameters as well as the fit fractions.

.. code-block:: python

   from tf_pwa.utils import error_print
   print("########## fitting parameters:")
   for key, value in config.get_params().items():
      print(key, error_print(value, errors.get(key, None)))
   print("########## fit fractions:")
   for i in fit_frac:
      print(i, " : " + error_print(fit_frac[i], err_frac.get(i, None)))

If the plotting options are also provided in the :file:`config_sample.yml`,
we can also plot the distributions of variables indicated in the configuration file.

.. code-block:: python

   config.plot_partial_wave(fit_result, prefix="figure/")

The figures will be saved under path :file:`figure`. Here are the three invariant mass pairs for example.

(three pictures here)

We can do a lot more using `tf_pwa`. For more examples, please see path :file:`tutorials`.





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
