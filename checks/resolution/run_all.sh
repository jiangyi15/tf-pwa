#!/usr/bin/bash

if [[ ! -e data ]];
then
    mkdir data
fi


python gen_toy.py
python gauss_sample.py

python ../../fit.py
python plot_resolution.py
