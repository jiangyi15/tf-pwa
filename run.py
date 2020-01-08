#!/usr/bin/python3
import os
import sys
import subprocess

def run(string):
    os.system(string)

#i=0
#aa='python run_sp.py > log{i}_sp.txt 2>&1'.format(i=0)
cmd = 'python fit_scipy.py > log.txt 2>&1; ./state_cache.sh'
for i in range(10):
    run(cmd)

#with open("log.txt","w") as file:
#    subprocess.run(args=["python","fit_scipy.py"],stdout=file,stderr=subprocess.STDOUT)

