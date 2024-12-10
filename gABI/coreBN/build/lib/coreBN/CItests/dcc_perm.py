import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects import default_converter


# import R's utility packages

lib_loc_Ulysses = "/home/sdigioia/R/lib64/R/library"

base = importr('base', lib_loc= lib_loc_Ulysses)
utils = importr('utils', lib_loc= lib_loc_Ulysses)
kpcalg = importr('kpcalg', lib_loc= lib_loc_Ulysses)
energy = importr('energy', lib_loc= lib_loc_Ulysses)



ro.r('''
	#create function wrapping dcov-test
	dcov.perm.wrapper <- function(x, # first variable
                       y, # second variable
                       index=1, #
		               p=200
                       ){
		pval = dcov.test(x,y, index=index, R=p)$p.value
		return(pval)
	}
	# end function
	''')



dcov_perm= ro.r['dcov.perm.wrapper']


np_cv_ru = default_converter + numpy2ri.converter

def Dcov_perm_or(x_arr, y_arr, index=1, perm=200):
    with np_cv_ru:
        pvalue = dcov_perm(x_arr, y_arr, index, perm)
    return pvalue
