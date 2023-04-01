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




#base = importr('base', lib_loc="/usr/lib/R/library")
#utils = importr('utils', lib_loc="/usr/lib/R/library")
#kpcalg = importr('kpcalg', lib_loc="/home/serafina/R/x86_64-pc-linux-gnu-library/4.2/")
#energy = importr('energy', lib_loc="/home/serafina/R/x86_64-pc-linux-gnu-library/4.2/")



ro.r('''
	#create function wrapping hsic-perm
	hsic.perm.wrapper <- function(x, # first variable
                       y, # second variable
                       sig=1, #sigma for the Gaussian kernel
		               p=100, #number of permutations
                       numCol=100 #cols Cholesky
                       ){
		pval = hsic.perm(x,y,sig=sig, p=p, numCol=numCol)$p.value
		return(pval)
	}
	# end function
	''')


hsic_p_wrapped= ro.r['hsic.perm.wrapper']


np_cv_ru = default_converter + numpy2ri.converter



def Hsic_perm_or(res_x, res_y, sigma=1, perm=100, numcol=100):
    numcol = round(len(res_x)/20)*10
    with np_cv_ru:
        pvalue = hsic_p_wrapped(res_x, res_y, sigma, perm, numcol)
    return pvalue
