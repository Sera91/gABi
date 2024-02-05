import logging
import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from hsic_gamma import Hsic_gamma_or
from hsic_perm import Hsic_perm_or
from dcc_perm import Dcov_perm_or
from dcc_gamma import Dcov_gamma_or

def kernel_CItest( x, y, list_set, data, verbose=False, method=None, p=None, index=None, sig=None, numCol=None, boolean=False, **kwargs):
    
        """
        This function tests whether x and y are conditionally independent given the subset S of the remaining nodes,
        using the independence criterions: Distance Covariance/HSIC
        It takes as arguments:
                 @dataframe :  data
                 @str       : x,y (identify vars over whic we test CI, in the dataframe)
                 @list of str: list_set  (list of names identifying vars in the conditioning set)
                 @bool param:  verbose (a logical parameter, if None it is setted to False. When True the detailed output is provided).
                 @str  param: method  (Method for the conditional independence test: Distance Covariance (permutation or gamma test), HSIC (permutation or gamma test) or HSIC cluster)
                 @int  param: p  (number of permutations for Distance Covariance, HSIC permutation and HSIC cluster tests)
                 @int  param: index (power index in (0,2]  for te formula of the distance in the Distance Covariance)
                 @float param: sig (Gaussian kernel width for HSIC tests. Default is 1)
                 @int  param: numCol (number of columns used in the incomplete Cholesky decomposition. Default is 100)
                 @int  param: numCluster (number of clusters for kPC clust algorithm)
                 @float param: eps   (Normalization parameter for kPC clust. Default is 0.1)
                 @int  param: paral (number of cores to use for parallel calculations.)
 		         boolean: bool
                 If boolean=True, an additional argument `significance_level` must
                 be specified. If p_value of the test is greater than equal to
                 `significance_level`, returns True. Otherwise returns False.
                If boolean=False, returns the pearson correlation coefficient and p_value
                of the test.         
        """
        x_arr = (data[x]).to_numpy()
        y_arr = (data[y]).to_numpy()
                 
                 
        if (method==None):
            method = 'hsic.gamma'
            print("Independence criterion method was not provided, using the default method: hsic-gamma")
        if (p==None):
                p = 100
                print("Number of perm not provided. Default is 100")
        if (index==None):
                index=2
                print("index for Dcov not provided. default is 2")
        if (sig==None):
                sig=1
                print("Gaussian kernel width for HSIC tests not provided. Default is 1")
        if (numCol==None):
                print("Number of cols to consider in Cholesky decomposition not provided. Default is 100")
                numCol=100
        #p_value=0.0
        if (len(list_set)==0):
                    print("pure independence test")
                    if (method=='dcc.perm'):
                        p_value = Dcov_perm_or(x_arr, y_arr, index, p)
                    elif (method=='dcc.gamma'):
                        p_value = Dcov_gamma_or(x_arr, y_arr, index)
                    elif (method=='hsic.perm'):
                        p_value = Hsic_perm_or(x_arr, y_arr,  sig, p, numCol)
                    elif (method=='hsic.gamma'):
                        p_value = Hsic_gamma_or(x_arr, y_arr,  sig)

                    else:
                        print("method should be one of hsic.gamma, hsic.perm, dcc.gamma")
                        return None

                    if boolean:
                        if p_value >= kwargs["significance_level"]:
                            return True
                        else:
                            return False
                    else:
                        return p_value
                    
        else :
                    data_Sset = (data[list_set]).to_numpy()
                    gam  = LinearGAM(np.sum([s(ii) for ii in range(len(list_set))]))
                    gam1 = LinearGAM(np.sum([s(ii) for ii in range(len(list_set))]))
                    gam.gridsearch(data_Sset,x_arr)
                    res_X = gam.deviance_residuals(data_Sset,x_arr)
                    gam1.gridsearch(data_Sset,y_arr)
                    res_Y = gam1.deviance_residuals(data_Sset,y_arr)
                         
                    if (method=='dcc.perm'):
                       p_value = Dcov_perm_or(res_X, res_Y, index, p)
                    elif (method=='dcc.gamma'):
                       p_value = Dcov_gamma_or(res_X, res_Y, index)
                    elif (method=='hsic.perm'):
                       p_value = Hsic_perm_or(res_X, res_Y,  sig, p, numCol)
                    elif (method=='hsic.gamma'):
                       p_value = Hsic_gamma_or(res_X, res_Y , sig)
                    else:
                       print("method should be one of hsic.gamma, hsic.perm, dcc.gamma")
                       return None
                    if boolean:
                        if p_value >= kwargs["significance_level"]:
                            return True
                        else:
                            return False
                    else:
                        return p_value
