gABiC package requires a Python version >= 3.10
and the Python packages listed in the README.md

To install in a safe and automatic way all the Python dependencies of gABiC
you can create a Python environment from the gABi.yml file, using conda,
as shown below:

```
$ conda env create -f gABI.yml
```

Then you can install the core of the gABiC package, doing:

```
$ cd gABiC/coreBN
```

```
$ python setup.py install
```





