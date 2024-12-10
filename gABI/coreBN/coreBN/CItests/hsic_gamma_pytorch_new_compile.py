#Hilber Schmidt Independence Criterion gamma test
#'
#' Test to check the independence between two variables x and y using HSIC.
#' The hsic.gamma() function, uses Hilbert-Schmidt independence criterion to test for independence between
#' random variables.
#'

#' @param x data of first sample
#' @param y data of second sample
#' @param sig inverse Width of Gaussian kernel. Default is 1
#' @param numCol maximum number of columns that we use for the incomplete Cholesky decomposition


import numpy as np
from math import factorial, sqrt, exp, ceil, pow
from itertools import combinations
from scipy.stats import gamma
from numba import njit
#from numba import cuda
#import cupy as cu
from scipy.sparse.linalg import svds
from sklearn.gaussian_process.kernels import RBF
import torch
from torch.utils.dlpack import from_dlpack
import gc




def rSubset(n, r):
  
    # return list of all subsets of length r
    # to deal with duplicate subsets use 
    # set(list(combinations(arr, r)))
    return list(combinations(np.arange(n), r))

@torch.compile
def variance_opt(K, L, N_obs):
    varHSIC = torch.pow((1.0/6.0 * torch.multiply(K, L)),2.0)
    Var_HSIC = 1.0/N_obs/(N_obs-1)* ( torch.sum(varHSIC) - torch.trace(varHSIC)) 
    Var_HSIC = ((72.0*(N_obs-4)*(N_obs-5)/N_obs/(N_obs-1)/(N_obs-2)/(N_obs-3))  * Var_HSIC).to('cpu').numpy()
    return Var_HSIC

@torch.compile
def mu_opt(N, T, M):
    mu = ((1.0/N/(N-1))*torch.sum(T - torch.diag(T)*M)).to('cpu').numpy()
    return mu

@torch.compile
def Hsic_teststat(K, L, N):
    return (torch.sum(torch.multiply(torch.t(K), L))).to('cpu').numpy()/(1.0*N*N)
    

@njit()
def CPU_kernel_matrix(x,n, sigma=1.0):
    """
    this function build the kernel matrix
    """
    out = np.ones((n,n), dtype=np.float64)
    for i in range(n):
        for j in range(i+1,n):
                out[i,j] = exp(-sigma*(x[i] - x[j])*(x[i]-x[j]))
                #print("K(",i,j,")=", out[i,j])
                out[j,i] = out[i,j]    
    return out


@njit()
def CPU_kernel_matrix_new(x,n):
    """
    this function build the kernel matrix on CPU
    with the kernel width given by the median pair distance
    """
    out = np.ones((n,n), dtype=np.float64)
    for i in range(n):
        for j in range(i+1,n):
                out[i,j] = (x[i] - x[j])*(x[i]-x[j])
                #print("K(",i,j,")=", out[i,j])
                out[j,i] = out[i,j]
    sigma_median = np.median(out[np.triu_indices(n,1)].flatten())
    out = np.exp(- out/2.0*sigma_median)
    return out




def kernel_matrix_scipy(x_arr, sig=1.0):
    """rbf kernel depend on the l param, that is related to the sigma of rbfdot
	from the formula sigma = 1/2 * l**2 """
    l = 1.0 / (sqrt(2.0*sig))
    kernel = 1.0*RBF(l)
    return kernel.__call__(x_arr.reshape(-1,1))


def svd_wrapper(M,k_input=100):
    u_K, s_K, vT_K= svds(M, k=k_input, which='LM') 
    return u_K, s_K, vT_K


def H_matrix_cpu(n):
    H = np.eye(n) - (1.0/n)*np.ones((n,n))
    return H

@torch.compile
def H_matrix(n, input_device, tensor_type):  
    #H = cu.eye(n,n) - (1.0/n)*cu.ones((n,n))
	H = torch.eye(n, dtype= tensor_type, device=input_device) - (1.0/n)*torch.ones(n,n, dtype=tensor_type, device=input_device)
	return H

def coeff_tuple(n,k):
    if k<n:
       return factorial(n)/factorial(n-k)
    else:
       print("Error: k should be less than n")
       return None


def trace_matrix(M):
    return torch.sum(torch.diag(M))

if torch.cuda.is_available():
    import cupy as cu
    from numba import cuda
    @cuda.jit
    def GPU_distance_matrix(x, n, out, sigma):
            i, j = cuda.grid(2)
            if (i < n) and ((i+1) <=j < n) :
                out[i, j] = (x[i]-x[j])*(x[i]-x[j])
                out[j, i] = out[i, j]  

    @cuda.jit
    def GPU_kernel_matrix(x, n, out, sigma):
            i, j = cuda.grid(2)
            if (i < n) and ((i+1) <=j < n) :
                out[i, j] = exp(-sigma*(x[i]-x[j])*(x[i]-x[j]))
                out[j, i] = out[i, j]    


def Hsic_gamma_py_new2(x, y, n_device, sigma=1.0, debug=False, gpu_selected=True):
    
    #' @details Let x and y be two samples of length n. 
    #Gram matrices K and L are defined as: K_{i,j} =exp(-sig^2 (x_i-x_j)^2)   (\eqn{K_{i,j} = \exp{- \sigma (x_i-x_j)^2}})
    #                                      L_{i,j} =exp(-sig^2 (y_i-y_j)^2)}. 
    #Defining H Matrix with elements : \eqn{H_{i,j} = \delta_{i,j} - \frac{1}{n}. 
    #Let \eqn{A=HKH} and \eqn{B=HLH}, then \eqn{HSIC(x,y)=\frac{1}{n^2}Tr(AB)}. 
    #Gamma test compares HSIC(x,y) with the alpha quantile of the gamma distribution with mean and variance such as HSIC under independence hypothesis.

    #tensor_device = torch.device("cpu")
    if (len(x) != len(y)):
        print("ERROR: the sample size of arrays is different!!")
        return 0
	
    #if (index < 0 or index > 2):
    #    print("ERROR: index must be in the interval [0,2)!")
    #    return 0
    if not(np.isfinite(x).all()): 
        print("Data contains missing or infinite values")
        return 0
    if not(np.isfinite(y).all()): 
        print("Data contains missing or infinite values")
        return 0
    n = len(x)

    cuda_string= "cuda:"+str(n_device)

    if torch.cuda.is_available() and gpu_selected:
        go_gpu=True
    else :
        go_gpu=False

    N_obs=len(x)

    tensor_device = torch.device(cuda_string if go_gpu else "cpu")
    tensor_type = torch.float64
    if (tensor_type == torch.float32):
        string_Ttype = 'float32'
    elif (tensor_type == torch.float64):
        string_Ttype = 'float64'

    
    if go_gpu:
        dev = cu.cuda.Device(n_device)
        with dev.use():
            gpu_x_arr = cu.asarray(x, dtype=string_Ttype)
            gpu_y_arr = cu.asarray(y, dtype=string_Ttype)
            threadsperblock = (16, 16)
            blockspergrid_x = ceil(N_obs / threadsperblock[0])
            blockspergrid_y = ceil(N_obs / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            K = cu.ones((N_obs, N_obs), dtype=string_Ttype)
            GPU_distance_matrix[blockspergrid, threadsperblock](gpu_x_arr,N_obs,K, 1.0)
            sigma_median_K = cu.median(cu.ravel(K[cu.triu_indices(K.shape[0],1)]))
            K = cu.exp(-K/(2.0*sigma_median_K))
            L = cu.ones((N_obs, N_obs), dtype=string_Ttype)
            GPU_distance_matrix[blockspergrid, threadsperblock](gpu_y_arr,N_obs,L, 1.0)
            sigma_median_L = cu.median(cu.ravel(L[cu.triu_indices(L.shape[0],1)]))
            L = cu.exp(-L/(2.0*sigma_median_L))
            del  gpu_x_arr, gpu_y_arr
            tensor_K = from_dlpack(K.toDlpack())
            tensor_K = tensor_K.type(tensor_type)
            tensor_L = from_dlpack(L.toDlpack())
            tensor_L = tensor_L.type(tensor_type)
            del K, L
            torch.cuda.empty_cache()
            
    else:
        #building Kernel matrix (using RBF gaussian kernel)
        kernel_K = CPU_kernel_matrix_new(x,N_obs, sigma=1.0)
        tensor_K = torch.from_numpy(kernel_K.astype(string_Ttype)).to(tensor_device)
        kernel_L = CPU_kernel_matrix_new(y,N_obs, sigma=1.0)
        tensor_L = torch.from_numpy(kernel_L.astype(string_Ttype)).to(tensor_device)
        del kernel_K, kernel_L

    
    #H_mat = H_matrix_cpu(N_obs)
    
    #H = torch.from_numpy(H_mat.astype(string_Ttype)).to(tensor_device)
    #Gram centered matrices
    
    M_unit = torch.eye(N_obs, device=tensor_device)
    mu_K = mu_opt(N_obs,tensor_K,M_unit)
    mu_L = mu_opt(N_obs,tensor_L,M_unit)
    H = H_matrix(N_obs,tensor_device,tensor_type)
    Kc = torch.mm(torch.mm(H,tensor_K), H) #Note: these are slightly biased estimates of centred Gram matrices
    Lc = torch.mm(torch.mm(H,tensor_L), H)
    Var_HSIC = variance_opt(Kc, Lc, N_obs)

    #HSIC_testStat = torch.sum(torch.multiply(torch.t(Kc), Lc))/N_obs
    #if debug:
    #    print("hsic", HSIC_testStat)
    HSIC_testStat=HSIC_testStat(Kd, Lc, N_obs)
        
    mean_HSIC = 1./N_obs * (1.0 + mu_K*mu_L - mu_K - mu_L ) # meand under H0
    
    
    del tensor_K, tensor_L, Kc, Lc, H
    gc.collect()
    if go_gpu:
        torch.cuda.empty_cache()

    
    alpha = pow(mean_HSIC,2.0)/ Var_HSIC
    beta = Var_HSIC/mean_HSIC

    p_value = 1 - gamma.cdf(x=HSIC_testStat, a =alpha, scale=beta) 
    
    if debug:
        print("variance HSIC:", Var_HSIC)
        print("pval:", p_value)
        print("hsic stat :", HSIC_testStat)
        print("mux:", mu_K)
        print("muy:", mu_L)
        print("mean HSIC:", mean_HSIC)

    return p_value
