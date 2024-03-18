from .utils_RCIT import pgamma, checkCoeffsArePositiveError, checkXvaluesArePositiveError
import numpy as np

def c(x,y,z):
    return np.array([x,y,z])

def hbe(coeff,x):
    """
    This Hall-Buckley-Eagleson (HBE) method computes the cdf of a positively-weighted sum of 
    chi-squared random variables.

    It takes as input the coeff (coefficient vector) and x (a vector).
    """
    if(coeff.any() == None or x == None):
        print("missing an argument - need to specify \"coeff\" and \"x\"")
        return None

    if (checkCoeffsArePositiveError(coeff)):
        exit(getCoeffError(coeff))

    if (checkXvaluesArePositiveError([x])):
        exit(getXvaluesError(x))

    kappa = c(np.sum(coeff), 2*np.sum(coeff**2), 8*np.sum(coeff**3))
    K_1 = np.sum(coeff)
    K_2 = 2 * np.sum(coeff**2)
    K_3 = 8 * np.sum(coeff**3)
    nu = 8 * (K_2**3)/(K_3**2)

    #gamma parameters for chi-square
    gamma_k = nu/2
    gamma_theta = 2

    #need to transform the actual x value to x_chisqnu ~ chi^2(nu)
	#This transformation is used to match the first three moments
	#First x is normalised and then scaled to be x_chisqnu
    x_chisqnu_vec = np.sqrt(2 * nu / K_2) * (x - K_1) + nu
	#now this is a chi_sq(nu) variable
    p_chisqnu_vec = pgamma(x_chisqnu_vec, shape=gamma_k, rate=gamma_theta)
    return p_chisqnu_vec
