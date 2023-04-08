import logging
import numpy as np
import sys
import gc
from pygam import LinearGAM, s
from .hsic_gamma_pytorch import Hsic_gamma_py
#from .hsic_perm import Hsic_perm_or
#from .dcc_perm import Dcov_perm_or
from .dcc_gamma_pytorch import Dcov_gamma_py
#import dask.dataframe as dd
from coreBN.utils import GAM_residuals, GAM_residuals_fast
import sys
#from dask.distributed import Client

def kernel_CItest_cycle( x, y, sep_sets, l_m , alpha, n_device, data, method='dcc.gamma', verbose=False):

        """
        This function, implemented for the parallel run of kpc, tests whether x and y are conditionally independent given all the subsets S, unique combination of the remaining nodes, inside the neighborood of x and y,
        using two independence criterions: Distance Covariance/HSIC
        It takes as arguments:
                 
                 @str       : x,y (identify vars over whic we test CI, in the dataframe)
                 @list of Ssets: list of separating sets, containing each a list of names identifying vars in the conditioning set
                 @l_m       : size of each subset
                 @dataframe :  data
                 @float param: alpha (significance level to test with the p-value test)
                 @int param : integer identifying the cuda device over which perform the GPU calculations
                 @str  param: method  (Method for the conditional independence test: Distance Covariance (permutation or gamma test), HSIC (permutation or gamma test) or HSIC cluster)
                 @int  param: p  (number of permutations for Distance Covariance, HSIC permutation and HSIC cluster tests)
                 @int  param: index (power index in (0,2]  for te formula of the distance in the Distance Covariance)
                 @float param: sig (Gaussian kernel width for HSIC tests. Default is 1)
                 @int  param: numCol (number of columns used in the incomplete Cholesky decomposition. Default is 100)
                 @bool param:  verbose (a logical parameter, if None it is setted to False. When True the detailed output is provided)
                 
        The function returns the  p_value and the corresponding sep_set         
        """


        from operator import itemgetter

        #print(" I am inside kernel function", flush=True)

        if (method =='dcc.gamma'):
            #print("selected method:dcc", flush=True)
            from coreBN.CItests import Dcov_gamma_py as Itest
        elif (method == 'hsic.gamma'):
            from coreBN.CItests import Hsic_gamma_py as Itest
        else:
            print("wrong method")
            sys.exit()


        l_sets = list(sep_sets)
        if verbose:
            print("first separating set", l_sets[0])
                 
        
        if (l_m<1):
            if verbose:
                print("pure independence test")
            final_x_arr = (data[x]).to_numpy()
            final_y_arr = (data[y]).to_numpy()
            p_value = Itest(final_x_arr, final_y_arr, n_device)
            del final_x_arr, final_y_arr
            gc.collect()
            return (x,y), {()}, p_value
        
        N_sets = len(l_sets)
        list_vars = data.columns.to_list()
        dict_ind = { list_vars[i]:i for i in range(0, len(list_vars) ) }
        #x_index = dict_ind[x]
        #y_index = dict_ind[y] 
        data_matrix = data.to_numpy()
        x_arr = data_matrix[:, dict_ind[x]]
        y_arr = data_matrix[:, dict_ind[y]]
        if (l_m==1):
            Si_sets = [dict_ind[list(sep)[0]] for sep in l_sets]
        else:
            Si_sets = [list(itemgetter(*sep_set)(dict_ind)) for sep_set in sep_sets]
        #print("conditioning sets:", sep_sets, flush=True)   
        
        if verbose:
            print("conditioning sets of len:",N_sets, sep_sets, flush=True)   
        del dict_ind
        for i in np.arange(0, N_sets):
                data_Sset= data_matrix[:, Si_sets[i]]
                res_x = GAM_residuals_fast(data_Sset, x_arr, l_m ) 
                res_y = GAM_residuals_fast(data_Sset, y_arr, l_m )
                p_value = Itest(res_x, res_y, n_device)  
                #p_value = Dcov_gamma_py_gpu(final_x_arr, final_y_arr, index)
                if (p_value > alpha) :
                    if verbose:
                        print(i,'-th pval:', p_value)
                    final_set = l_sets[i]
                    del data_Sset, data_matrix, x_arr, y_arr, Si_sets, list_vars, l_sets, res_x, res_y
                    gc.collect()
                    return (x,y), final_set, p_value
        final_set = l_sets[(N_sets-1)]
        del data_Sset, data_matrix, x_arr, y_arr, Si_sets, list_vars, l_sets, res_x, res_y
        gc.collect()
        return (x,y), final_set , p_value

        

def kernel_CItest( x, y, list_set, data, method='hsic.gamma', p=None, index=1, sig=1, numCol=None, verbose=False, boolean=False, **kwargs):
    
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

        #if dask_cluster!=None:
        #    client = Client(dask_cluster)
        if verbose:
        	print("X variable: ", x)
        	print("Y variable:", y)

        #data = pd.read_csv(data_input)
        #reading parquet files
        #ddf = dd.read_parquet(data_input)
        #all_vars = list(list_set)
        #all_vars.append(x)
        #all_vars.append(y)

        #data = ddf[all_vars].compute()
        #data = data.head(1000)
        x_arr = (data[x]).to_numpy()
        y_arr = (data[y]).to_numpy()
                 
        #if(boolean==True):
        #    print("significance level:", kwargs["significance_level"])    
        #if debug:
        #    print("Independence criterion method was not provided, using the default method: hsic-gamma")
        if (p==None):
                p = 100
                #print("Number of perm not provided. Default is 100")
        #if (index==None):
        #        index=1
                #print("index for Dcov not provided. default is 1")
        #if (sig==None):
        #        sig=1
                #print("Gaussian kernel width for HSIC tests not provided. Default is 1")
        if (numCol==None):
                #print("Number of cols to consider in Cholesky decomposition not provided. Default is 100")
                numCol=100

        #p_value=0.0
        N_cond_vars = len(list_set)
        if (N_cond_vars<1):
                    if verbose:
                       print("pure independence test")
                    final_x_arr = x_arr
                    final_y_arr = y_arr
        else :
                    
                    list_set = list(list_set)
                    if verbose:
                       print("list vars in conditioning set:", list_set)
                    print(type(list_set))
                    data_Sset = (data[list_set]).to_numpy()
                    gam  = LinearGAM(np.sum([s(ii) for ii in range(N_cond_vars)]))
                    gam1 = LinearGAM(np.sum([s(ii) for ii in range(N_cond_vars)]))
                    gam.gridsearch(data_Sset,x_arr)
                    res_X = gam.deviance_residuals(data_Sset,x_arr)
                    gam1.gridsearch(data_Sset,y_arr)
                    res_Y = gam1.deviance_residuals(data_Sset,y_arr)
                    final_x_arr = res_X
                    final_y_arr = res_Y
                    del data_Sset
                         
                    
        #match method:
        #    case 'dcc.perm':    
                  
        if method =='dcc.gamma':    
                  #NEED  to introduce flag for backend here
                  #p_value = Dcov_gamma_py_gpu(final_x_arr, final_y_arr, index)
                  p_value = Dcov_gamma_py(final_x_arr, final_y_arr, 0)
        #elif method == 'dcc.perm':
        #          p_value = Dcov_perm_or(final_x_arr, final_y_arr, index, p)
        #elif method=='hsic.perm':
        #          p_value = Hsic_perm_or(final_x_arr, final_y_arr,  sig, p, numCol)
        elif method == 'hsic.gamma':
                  p_value = Hsic_gamma_py(final_x_arr, final_y_arr, 0)
        else:
            sys.exit()
        
        
        if verbose:
           print('pval:', p_value)

        if boolean:
                  if (p_value >= kwargs["significance_level"]):
                        print('edge ', x+'-'+y,'pval:', p_value)
                        return True
                  else:
                        return False
        else:
                  return p_value
