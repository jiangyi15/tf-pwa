Setup for Developer Enveriment
------------------------------

.. note::
   Before the developing, creating a standalone enveriment is recommanded (see https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands for more).


The main steps are similar as normal install, only two extra things need to be done.

The first things is writing tests, and tests your code.
We use pytest (https://docs.pytest.org/en/stable/) framework, You should install it.

.. code::

    conda install pytest pytest-cov pytest-benchmark


The other things is `pre-commit`. it need for developing.

1. You can install `pre-commit` as

.. code::

    conda install pre-commit

and

2. then enable `pre-commit` in the source dir

.. code::

    conda install pylint # local dependences
    pre-commit install

You can check if pre-commit is working well by running

.. code::

    pre-commit run -a

It may take some time to install required package.

.. note::
   If there are some `GLIBC_XXX` errors at this step, you can try to install `node.js`.

.. note::
   For developer using editor with formatter, you should be careful for the options.

The following are all commands needed

.. code::

    # create environment
    conda create -n tfpwa2 python=3.7 -y
    conda activate tfpwa2

    # install tf-pwa
    conda install --file requirements-min.txt -y
    python -m pip install -e . --no-deps
    # install pytest
    conda install pytest pytest-cov -y
    # install pylint local
    conda install pylint
    # install pre-commit
    conda install pre-commit -c conda-forge -y
    pre-commit install
