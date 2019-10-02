#!/bin/bash

cd ./bayesfast/modules/
python setup.py build_ext --inplace
cd ../../bayesfast/transforms/
python setup.py build_ext --inplace
cd ../../bayesfast/utils/
python setup.py build_ext --inplace
cd ../../cosmofast/planck2018/
python setup.py build_ext --inplace
