
# A Partial Wave Analysis program using Tensorflow



[![pipeline status](https://gitlab.com/jiangyi15/tf-pwa/badges/dev/pipeline.svg)](https://gitlab.com/jiangyi15/tf-pwa/-/commits/dev)
[![coverage report](https://gitlab.com/jiangyi15/tf-pwa/badges/dev/coverage.svg)](https://gitlab.com/jiangyi15/tf-pwa/-/commits/dev)


## Install 

Get the packages using ```git clone```

### conda

get miniconda for python3 from [minicoda](https://docs.conda.io/en/latest/miniconda.html) and install it.

install following packages

```
conda install tensorflow-gpu pyyaml sympy matplotlib scipy
```

### pip

```
pip3 install -r requirements.txt
python3 setup.py install
```

## Scripts

### fit.py

simple fit scripts, 
decay structure is described in ```config.yml```

```
python fit.py
```
fit parameters will save in final_params.json,
figure can be found in ```figure/```


### state_cache.sh

script for cache state, using the latest *_params.json file as parameters and cache newer files in ```path``` (the default is ```trash/```).

```
./state_cache.sh [path]
```

## Documents

Autodoc using sphinx-doc, need sphinx-doc 

```
  python setup.py build_sphinx 
```

## Dependencies

tensorflow or tensorflow-gpu >= 2.0.0 

sympy : symbolic expression

PyYAML : config.yml file

matplotlib : plot

scipy : fit

