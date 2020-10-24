
# A Partial Wave Analysis program using Tensorflow

[![Documentation build status](https://readthedocs.org/projects/tf-pwa/badge/?version=latest)](https://tf-pwa.readthedocs.io)
[![CI status](https://github.com/jiangyi15/tf-pwa/workflows/CI/badge.svg)](https://github.com/jiangyi15/tf-pwa/actions?query=branch%3Adev+workflow%3ACI)
[![Test coverage](https://codecov.io/gh/jiangyi15/tf-pwa/branch/dev/graph/badge.svg)](https://codecov.io/gh/jiangyi15/tf-pwa)
[![conda cloud](https://anaconda.org/jiangyi15/tf-pwa/badges/version.svg)](https://anaconda.org/jiangyi15/tf-pwa)
[![license](https://anaconda.org/jiangyi15/tf-pwa/badges/license.svg)](https://choosealicense.com/licenses/mit/)
<br>
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is a package and application for partial wave analysis (PWA) using TensorFlow.
By using simple configuration file (and some scripts), PWA can be done fast and automatically.



## Install

Get the packages using

```
git clone https://gitlab.com/jiangyi15/tf-pwa
```


The dependencies can be installed by `conda` or `pip`.

### conda (recommended)

When using conda, you don't need to install CUDA for TensorFlow specially.

  1. Get miniconda for python3 from [miniconda3](https://docs.conda.io/en/latest/miniconda.html) and install it.

  2. Install following packages

```
conda install tensorflow-gpu pyyaml sympy matplotlib scipy
```

  3. The following command can be used to set environment variables of Python. (Use `--no-deps` to make sure that no PyPI package will be installed)

```
python -m pip install -e . --no-deps
```

  4. (option) There are some option packages, such as `uproot` for reading root file. It can be installed as

```
conda install uproot -c conda-forge
```

### conda channel (experimental)

A pre-built conda package (Linux only) is also provided, just run following command to install it.

```
conda config --add channels jiangyi15
conda install tf-pwa
```

### pip

When using `pip`, you will need to install CUDA to use GPU. Just run the following command :

```bash
python3 -m pip install -e .
```

To contribute to the project, please also install additional developer tools with:

```bash
python3 -m pip install -e .[dev]
```


## Scripts

### fit.py

simple fit scripts,
decay structure is described in ```config.yml```

```
python fit.py [--config config.yml]  [--init_params init_params.json]
```

fit parameters will save in final_params.json,
figure can be found in ```figure/```


### state_cache.sh

script for cache state, using the latest *_params.json file as parameters and cache newer files in ```path``` (the default is ```trash/```).

```
./state_cache.sh [path]
```

## Documents

See [https://jiangyi15.gitlab.io/tf-pwa/](https://jiangyi15.gitlab.io/tf-pwa/) for more information.

Autodoc using sphinx-doc, need sphinx-doc

```
python setup.py build_sphinx
```

Then, the documents can be found in build/sphinx/index.html.

Documents cna also build with `Makefile` in `docs` as

```
cd docs && make html
```
Then, the documents can be found in docs/_build/html.

## Dependencies

tensorflow or tensorflow-gpu >= 2.0.0

sympy : symbolic expression

PyYAML : config.yml file

matplotlib : plot

scipy : fit

