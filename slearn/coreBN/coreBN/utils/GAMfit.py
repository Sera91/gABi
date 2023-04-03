import numpy as np
import pandas as pd
import sys
import os
import gc
from pygam import LinearGAM,s

def GAM_residuals(data_S, x_arr, N_Svars ):
    gam = LinearGAM(np.sum([s(ii) for ii in range(N_Svars)]))
    gam.gridsearch(data_S,x_arr)
    return gam.deviance_residuals(data_S,x_arr)

def GAM_residuals_fast(data_S, x_arr, N_Svars ):
    gam = LinearGAM(np.sum([s(ii) for ii in range(N_Svars)]))
    gam.fit(data_S,x_arr)
    return gam.deviance_residuals(data_S,x_arr)