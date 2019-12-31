#!/bin/bash

n=100

for i in {1..2}
do
  echo "Iterations ${i}:"
  time ./fit_scipy.py > fit_scipy.log
  ./plot_amp.py > plot_amp.log
  fcn=`tail fit_curve.json -n 3 | head -n 1`
  echo ${fcn} > FCN
  echo "FCN = ${fcn}"
  ./state_cache.sh
done
