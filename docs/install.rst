Install
=======

To use TFPWA, we need some dependent packages. There are two main ways,
`conda` and `virtualenv` you can choose *one* of them. *Or* you can choose other method in :ref:`other-method`

1. vitrual environment
----------------------

To avoid conflict of dependence, we recommed to use vitrual environment. there are two main vitrual environment for python packages,
`conda <https://conda.io/projects/conda/en/latest/index.html>`_ and  `virtualenv <https://virtualenv.pypa.io/en/latest/>`_. You can choose one of them. Since conda include cudatoolkit for gpu, we recommed it for user.


1.1 conda
`````````

- 1.1.1 Get miniconda for python3 from `miniconda3 <https://docs.conda.io/en/latest/miniconda.html>`_ and install it.

- 1.1.2 Create a virtual environment by

.. code::

   conda create -n tfpwa

, the `-n \<name\>` option will create a environment named by `\<name\>`. You can also use `-p \<path\>` option to create environment in the `\<path\>` directory.

- 1.1.3 You can activate the environment by

.. code::

   conda activate tfpwa

and then you can install packages in the conda environment

- 1.1.4 You can exit the environment by

.. code::

   conda deactivate

1.2 virtualenv
``````````````

- 1.2.1 You should have a python3 first.

- 1.2.2 Install virtualenv

.. code::

   python3 -m pip install --user virtualenv

- 1.2.3 Create a virtual environment

.. code::

   python3 -m virtualenv ./tfpwa

, it will store in the path `tfpwa`


- 1.2.4 You can activate the environment by

.. code::

   source ./tfpwa/bin/activete

- 1.2.5 You can  exit the environment by

.. code::

   deactivate


2. tensorflow2
--------------

The most important package is `tensorflow2 <https://github.com/tensorflow/tensorflow>`_.
We recommed to install tensorflow first. You can following the install instructions in `tensorflow website <https://tensorflow.google.cn/install>`_ (or `tensorflow.org <https://tensorflow.org/install>`_).

2.1 conda
`````````

Here we provide the simple way to install tensorflow2 gpu version in conda environment

.. code::

   conda install tensorflow-gpu=2.4

it will also install cudatoolkit.

2.2 virtualenv
``````````````

When using `virtualenv`, there is also simple way to install tensorflow2

.. code::

   python -m pip install tensorflow

, but you should check you CUDA installation for GPU.

.. note::

   You can use `-i https://pypi.tuna.tsinghua.edu.cn/simple` option to use pypi mirror site.


3. Other dependences
--------------------

Other dependences of TFPWA is simple.


3.1 Get TFPWA package
`````````````````````


Get the packages using

.. code::

   git clone https://github.com/jiangyi15/tf-pwa


3.2 conda
`````````

3.2.1 other dependences

In conda environment, go into the directory of `tf-pwa`, you can install the rest dependences by

.. code::

   conda install --file requirements-min.txt

.. note::
   we recommed Ampere card users to install with
   `tensorflow_2_6_requirements.txt` (see this
   `technical FAQ <https://tf-pwa.readthedocs.io/en/latest/tensorflow_version.html>`_).

   .. code::

      conda install --file tensorflow_2_6_requirements.txt -c conda-forge

3.2.2 TFPWA

install TFPWA

.. code::

   python -m pip install -e ./ --no-deps

Use `--no-deps` to make sure that no PyPI package will be installed.
Using `-e`, so it can be updated by `git pull` directly.


3.3 virtualenv
``````````````

In virtualenv, You can install dependences and TFPWA together.

.. code::

   python3 -m pip install -e ./

Using `-e`, so it can be updated by `git pull` directly.


4. (option)  Other dependences.
-------------------------------

   There are some option packages, such as `uproot` for reading root file.

4.1 conda
`````````

It can be installed as

.. code::

   conda install uproot -c conda-forge


4.2 virtualenv
``````````````
It can be installed as

.. code::

   python -m pip install uproot


.. _other-method:

5. Other install method.
------------------------

We also provided other install method.


5.1 conda channel (experimental)
````````````````````````````````

A pre-built conda package (Linux only) is also provided, just run following
command to install it.

.. code::

   conda config --add channels jiangyi15
   conda install tf-pwa

5.2  pip
````````

When using `pip`, you will need to install CUDA to use GPU. Just run the
following command :

.. code::

   python3 -m pip install -e .



6. For developer
----------------


To contribute to the project, please also install additional developer tools
with:

.. code::

   python3 -m pip install -e .[dev]
