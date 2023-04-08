coreBN
=====


coreBN is a python library for Automatic Structure Learning in the framework of Probabilistic (Causal) Graphical Models.


Dependencies
=============

It has following non optional dependencies:
- python 3.8
- networkX
- scipy 
- numpy
- pandas
- pytorch
- tqdm
- pyparsing
- statsmodels
- pickle
- joblib
- pyGAM
- Dask



Installation
============

cd coreBN/
$ pip install -r requirements.txt
$ python setup.py install

Testing
-------

After installation, you can launch the test from
source directory (you will need to have the ``pytest`` package installed):
```bash
$ pytest -v
```
to see the coverage of existing code use following command
```
$ pytest --cov-report html --cov=coreBN
```

Documentation and usage
=======================


I will use sphinx to build the documentation. To build the documentation on your local system use:
```
$ cd /docs
$ make html
```
The generated docs will be in _build/html


