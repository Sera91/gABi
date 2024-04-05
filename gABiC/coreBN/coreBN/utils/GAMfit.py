import numpy as np
import pandas as pd
import sys
import os
import gc
from pygam import LinearGAM,s

def GAM_residuals(data_S, x_arr, N_Svars ):
    """
    This function perform the GAM regression of the variable in x, and the variables in its conditioning set (S),
    and it outputs the residuals of the GAM regression.
    It is based on the pygam library. 
    From this library we call the functions:
    - LinearGAM to build the GAM model,
    - the function gridsearch to perform a gridsearch evaluation of the best model,
    - deviance_residuals to output the residuals of fitted data.
    """
    gam = LinearGAM(np.sum([s(ii) for ii in range(N_Svars)]))
    gam.gridsearch(data_S,x_arr)
    return gam.deviance_residuals(data_S,x_arr)

def GAM_residuals_fast(data_S, x_arr, N_Svars ):
    """
    This function perform the GAM regression of the variable in x, and the variables in its conditioning set (S),
    and it outputs the residuals of the GAM regression.
    It is based on the pygam library. 
    From this library we call the functions:
    - LinearGAM to build the GAM model,
    - the function fit to fix the parameters of the model fitting the data,
    - deviance_residuals to output the residuals of fitted data.
    """
    gam = LinearGAM(np.sum([s(ii) for ii in range(N_Svars)]))
    gam.fit(data_S,x_arr)
    return gam.deviance_residuals(data_S,x_arr)
