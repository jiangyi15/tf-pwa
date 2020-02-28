
# A Partial Wave Analysis program using Tensorflow



[![pipeline status](https://gitlab.com/jiangyi15/tf-pwa/badges/dev/pipeline.svg)](https://gitlab.com/jiangyi15/tf-pwa/-/commits/dev)
[![coverage report](https://gitlab.com/jiangyi15/tf-pwa/badges/dev/coverage.svg)](https://gitlab.com/jiangyi15/tf-pwa/-/commits/dev)


## Install 
### conda

get miniconda for python3 from [minicoda](https://docs.conda.io/en/latest/miniconda.html) and install it.

install following packages

```
conda install tensorflow iminuit sympy matplotlib scipy
```

then get the packages using ```git clone```

### pip

```
pip3 install -r requirements.txt
python3 setup.py install
```

## Scripts

### fit.py or fit_scipy.py 

simple fit scripts, 
Resonances is describing in ```Resonances.yml```

```
python fit.py
```
fit parameters will save in final_params.json

### plot_amp.py 

resonance plot scripts

```
python plot_amp.py
```

### fitfractions.py 

calculate fitfractions with error

```
python fitfractions.py
```

### state_cache.sh

script for cache state, using the latest *_params.json file as parameters and cache newer files in ```trash/```.

```
./state_cache.sh [path]
```

## autodocs

autodoc using sphinx-doc, need sphinx-doc 

```
  python setup.py build_sphinx 
```

## Dependencies

tensorflow or tensorflow-gpu >= 2.0.0 

sympy 

iminuit 

matplotlib : for plot_amp.py

scipy : for fit

