# Hilber Schmidt Independence Criterion gamma test
# '
# ' Test to check the independence between two variables x and y using HSIC.
# ' The hsic.gamma() function, uses Hilbert-Schmidt independence criterion to test for independence between
# ' random variables.
# '

# ' @param x data of first sample
# ' @param y data of second sample
# ' @param sig Width of Gaussian kernel. Default is 1
# ' @param numCol maximum number of columns that we use for the incomplete Cholesky decomposition
# '
# The original R function hsic.gamma() returns a list with class htest containing
# ' \item{method}{description of test}
# ' \item{statistic}{observed value of the test statistic}
# ' \item{estimate}{HSIC(x,y)}
# ' \item{estimates}{a vector: [HSIC(x,y), mean of HSIC(x,y), variance of HSIC(x,y)]}
# ' \item{replicates}{replicates of the test statistic}
# ' \item{p.value}{approximate p-value of the test}
# ' \item{data.name}{desciption of data}
# ' @author Petras Verbyla (\email{petras.verbyla@mrc-bsu.cam.ac.uk}) and Nina Ines Bertille Desgranges

import numpy as np
#from math import factorial
from itertools import combinations
#from scipy.stats import gamma
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter


# import R's utility packages

lib_loc_Ulysses = "/home/sdigioia/R/lib64/R/library"

base = importr('base', lib_loc= lib_loc_Ulysses)
utils = importr('utils', lib_loc= lib_loc_Ulysses)
kpcalg = importr('kpcalg', lib_loc= lib_loc_Ulysses)
energy = importr('energy', lib_loc= lib_loc_Ulysses)


#on Sera's laptop:
#base = importr('base', lib_loc="/usr/lib/R/library")
#utils = importr('utils', lib_loc="/usr/lib/R/library")
#kpcalg = importr('kpcalg', lib_loc="/home/serafina/R/x86_64-pc-linux-gnu-library/4.2/")
#energy = importr('energy', lib_loc="/home/serafina/R/x86_64-pc-linux-gnu-library/4.2/")


ro.r('''
	#create function wrapping hsic-gamma
	hsic.gamma.wrapper <- function(x, # first variable
                       y, # second variable
                       sigma=1, #sigma for the Gaussian kernel
                       numcol=100 #cols Cholesky decomposition
                       ){
		#print(sigma)
		print(x)
                #set.seed(4)
		pval = hsic.gamma(x,y,sig=sigma,numCol = numcol)$p.value
		return(pval)
	}
	# end function
	''')


hsic_g_wrapped = ro.r['hsic.gamma.wrapper']

np_cv_ru = default_converter + numpy2ri.converter


def Hsic_gamma_or(x_arr, y_arr, sigma=1):
    print("x_arr", x_arr)
    numcol = round(len(x_arr)/20)*10
    with np_cv_ru:
        pvalue = hsic_g_wrapped(x_arr, y_arr, sigma, numcol)
    return pvalue


