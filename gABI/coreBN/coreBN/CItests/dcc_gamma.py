import numpy as np
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



#on laptop Sera:
#base = importr('base', lib_loc="/usr/lib/R/library")
#utils = importr('utils', lib_loc="/usr/lib/R/library")
#kpcalg = importr('kpcalg', lib_loc="/home/serafina/R/x86_64-pc-linux-gnu-library/4.2/")
#energy = importr('energy', lib_loc="/home/serafina/R/x86_64-pc-linux-gnu-library/4.2/")

ro.r('''
	#create function wrapping dcov-test
	dcov.gamma.wrapper <- function(x, # first variable
                       y, # second variable
                       index=1, #
		               numCol=100
                       ){
     		pval=dcov.gamma(x, y, index = index, numCol = numCol)$p.value
		return(pval)
	}
	#end function 
	''')


dcov_gamma= ro.r['dcov.gamma.wrapper']


np_cv_ru = default_converter + numpy2ri.converter

def Dcov_gamma_or(x_arr, y_arr, index=1):
    numcol= round(len(x_arr)/20)*10
    with np_cv_ru:
        pvalue = dcov_gamma(x_arr, y_arr, index, numcol)
    return pvalue
