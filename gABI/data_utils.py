import numpy as np
import os
import wget
import zipfile




# %% Download data from github source
def download_example(data, verbose=3):
    # Set url location
    url = 'https://erdogant.github.io/datasets/'
    url=url + data + '.zip'

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    PATH_TO_DATA = os.path.join(curpath, wget.filename_from_url(url))
    if not os.path.isdir(curpath):
        os.mkdir(curpath)

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        if verbose>=3: print('[bnlearn] >Downloading example [%s] dataset..' %(data))
        wget.download(url, curpath)

    return PATH_TO_DATA
