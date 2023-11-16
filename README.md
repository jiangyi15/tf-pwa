# A Partial Wave Analysis program using Tensorflow

[![Documentation build status](https://readthedocs.org/projects/tf-pwa/badge/?version=latest)](https://tf-pwa.readthedocs.io)
[![CI status](https://github.com/jiangyi15/tf-pwa/workflows/CI/badge.svg)](https://github.com/jiangyi15/tf-pwa/actions?query=branch%3Adev+workflow%3ACI)
[![Test coverage](https://codecov.io/gh/jiangyi15/tf-pwa/branch/dev/graph/badge.svg)](https://codecov.io/gh/jiangyi15/tf-pwa)
[![conda cloud](https://anaconda.org/jiangyi15/tf-pwa/badges/version.svg)](https://anaconda.org/jiangyi15/tf-pwa)
[![pypi](https://img.shields.io/pypi/v/TFPWA)](https://pypi.org/project/TFPWA/)
[![license](https://anaconda.org/jiangyi15/tf-pwa/badges/license.svg)](https://choosealicense.com/licenses/mit/)
<br>
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jiangyi15/tf-pwa/HEAD)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)](https://github.com/pre-commit/pre-commit)
[![Prettier](https://camo.githubusercontent.com/687a8ae8d15f9409617d2cc5a30292a884f6813a/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f636f64655f7374796c652d70726574746965722d6666363962342e7376673f7374796c653d666c61742d737175617265)](https://prettier.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

This is a package and application for partial wave analysis (PWA) using
TensorFlow. By using simple configuration file (and some scripts), PWA can be
done fast and automatically.

## Install

You can go to
[http://tf-pwa.readthedocs.io/install](http://tf-pwa.readthedocs.io/en/latest/install.html)
for more informations. Get the packages using

```
git clone https://github.com/jiangyi15/tf-pwa.git
```

The dependencies can be installed by `conda` or `pip`.

### conda (recommended)

When using conda, you don't need to install CUDA for TensorFlow specially.

1. Get miniconda for python3 from
   [miniconda3](https://docs.conda.io/en/latest/miniconda.html) and install it.

2. Install requirements, we recommed Ampere card users to install with
   `tensorflow_2_6_requirements.txt` (see this
   [technical FAQ](https://tf-pwa.readthedocs.io/en/latest/tensorflow_version.html)).

You can install a tensorflow gpu version in anaconda as

```
conda install tensorflow[build="gpu*"]=2.8
```

and then install the rest dependences

```
conda install --file requirements-min.txt
```

**Or**

You can install a newer version in conda-forge as

```
conda install --file tensorflow_2_6_requirements.txt -c conda-forge
```

3. The following command can be used to set environment variables of Python.
   (Use `--no-deps` to make sure that no PyPI package will be installed. Using
   `-e`, so it can be updated by `git pull` directly.)

```
python -m pip install -e . --no-deps
```

4. (option) There are some option packages, such as `uproot` for reading root
   file. It can be installed as

```
conda install uproot -c conda-forge
```

<details><summary>
### conda channel (experimental)
</summary><p>

A pre-built conda package (Linux only) is also provided, just run following
command to install it.

```
conda config --add channels jiangyi15
conda install tf-pwa
```

</p></details>

<details><summary>
###  pip
</summary><p>
When using `pip`, you will need to install CUDA to use GPU. Just run the
following command :

```bash
python3 -m pip install -e .
```

To contribute to the project, please also install additional developer tools
with:

```bash
python3 -m pip install -e .[dev]
```

</p></details>

## Scripts

### fit.py

simple fit scripts, decay structure is described in `config.yml`, here `[]`
means options.

```
python fit.py [--config config.yml]  [--init_params init_params.json]
```

fit parameters will save in final_params.json, figure can be found in
`figure/`.

### state_cache.sh

script for cache state, using the latest \*\_params.json file as parameters and
cache newer files in `path` (the default is `trash/`).

```
./state_cache.sh [path]
```

## Documents

See [tf-pwa.rtfd.io](http://tf-pwa.readthedocs.io) for more information.

Autodoc using sphinx-doc, need sphinx-doc

```
python setup.py build_sphinx
```

Then, the documents can be found in build/sphinx/index.html.

Documents can also build with `Makefile` in `docs` as

```
cd docs && make html
```

Then, the documents can be found in docs/\_build/html.

## Other resources

[面向 BESIII 用户的 TF-PWA 使用手册](https://note.ihep.ac.cn/s/xAr0zQXf8)

[HADRON 2023](https://agenda.infn.it/event/33110/contributions/198135/)

[PWA 12](https://indico.cern.ch/event/885396/timetable/#52-reach-on-the-partial-wave-a)

## Dependencies

tensorflow or tensorflow-gpu >= 2.0.0

cudatoolkit : CUDA library for GPU acceleration

sympy : symbolic expression

PyYAML : config.yml file

matplotlib : plot

scipy : fit
