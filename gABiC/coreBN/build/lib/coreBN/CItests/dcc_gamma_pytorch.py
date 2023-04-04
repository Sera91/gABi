import numpy as np
from math import sqrt, exp, ceil, pow
from scipy.stats import gamma
import gc
import torch
from numba import njit 
import time


def H_matrix(n, input_device, tensor_type):  
    #H = cu.eye(n,n) - (1.0/n)*cu.ones((n,n))
	H = torch.eye(n, dtype= tensor_type, device=input_device) - (1.0/n)*torch.ones(n,n, dtype=tensor_type, device=input_device)
	return H




@njit()
def Eucl_matrix_L1_cpu(x,n):
    out = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        for j in range(i+1,n):
                out[i,j] = abs(x[i] - x[j])
                #print("K(",i,j,")=", out[i,j])
                out[j,i] = out[i,j]
    return out

if torch.cuda.is_available():
    import cupy as cu
    from numba import cuda
    from torch.utils.dlpack import from_dlpack
                
    @cuda.jit
    def distance_kernel_fast(x, n, m, out):
        i, j = cuda.grid(2)
        if (i < n) and ((i+1) <=j < m) :
            out[i, j] = abs(x[i] - x[j])
            out[j, i] = out[i, j]

def eigen_decomposition(input_tensor, k_sel=100, approx=False):
    if (approx==False):
        eigenvals, eigenvectors = torch.linalg.eigh(input_tensor, UPLO='U')
    else: 
        eigenvals, eigenvectors = torch.lobpcg(input_tensor, k=k_sel, largest=True, method="ortho")
        #we can use this method because since the matrix is real and symmetric it is positive definite
    return eigenvals, eigenvectors

	

def Dcov_gamma_py(x_arr, y_arr, n_device, index=1, gpu_selected=True, verbose=False):
    if (len(x_arr) != len(y_arr)):
        print("ERROR: the sample size of arrays is different!!")
        return 0
	
    #if (index < 0 or index > 2):
    #    print("ERROR: index must be in the interval [0,2)!")
    #    return 0
    if not(np.isfinite(x_arr).all()): 
        print("Data contains missing or infinite values")
        return 0
    if not(np.isfinite(y_arr).all()): 
        print("Data contains missing or infinite values")
        return 0
    n = len(x_arr)

    cuda_string= "cuda:"+str(n_device)

	#g_xarr= cu.asarray(x_arr)
	#g_yarr= cu.asarray(y_arr)

    tensor_type = torch.float64

    if (tensor_type == torch.float32):
        string_Ttype = 'float32'
        
    elif (tensor_type == torch.float64):
        string_Ttype = 'float64'

    if torch.cuda.is_available() and gpu_selected:
        go_gpu=True
    else :
        go_gpu=False

    #print("selected GPU: ", n_device)    

    tensor_device = torch.device(cuda_string if go_gpu else "cpu")

    
    if go_gpu:
        dev = cu.cuda.Device(n_device)
        with dev.use():
            threadsperblock = (16, 16)
            blockspergrid_x = ceil(n / threadsperblock[0])
            blockspergrid_y = ceil(n / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            gpu_kernel = cu.zeros((n, n), dtype='float64')
            #cu.cuda.runtime.deviceSynchronize()
            #startA = time.time()
            gpu_xarr = cu.asarray(x_arr)
            distance_kernel_fast[blockspergrid, threadsperblock](gpu_xarr, n, n, gpu_kernel)
            #cu.cuda.runtime.deviceSynchronize()
            #endA = time.time()	
            #print("time for kernel distance matrix on GPU:", endA-startA)
            tensor_X = from_dlpack(gpu_kernel.toDlpack())
            tensor_X = tensor_X.type(tensor_type)
            
            gpu_yarr = cu.asarray(y_arr) #cuda.to_device(y_arr)
            gpu_kernel_Y = cu.zeros((n, n), dtype='float64')
            distance_kernel_fast[blockspergrid, threadsperblock](gpu_yarr, n, n, gpu_kernel_Y)
            tensor_Y = from_dlpack(gpu_kernel_Y.toDlpack())
            tensor_Y = tensor_Y.type(tensor_type)
            del gpu_kernel, gpu_kernel_Y, gpu_yarr, gpu_xarr
            gc.collect()
            torch.cuda.empty_cache()
    else:
        kernel_X = Eucl_matrix_L1_cpu(x_arr,n)
        kernel_Y = Eucl_matrix_L1_cpu(y_arr,n)
        tensor_X = torch.from_numpy(kernel_X.astype(string_Ttype)).to(tensor_device)
        tensor_Y = torch.from_numpy(kernel_Y.astype(string_Ttype)).to(tensor_device)
        #end = time.time()
        #print("time for kernel distance matrix on CPU:", end-start) 

    
    #if (torch.equal(torch.t(tensor_X), tensor_X)):
    #            print("matrix is symmetric as expected")
    #if torch.allclose(tensor_X,tensor_X.T):
    #    print("kernel matrix is symmetric ...fiuu")

    #print("device of tensor X", tensor_X.device)

    avg_X = torch.mean(tensor_X)    
    eigenvals_X, U_X = torch.linalg.eigh(tensor_X, UPLO='U')   

    S_X = torch.diag(eigenvals_X.real)


    #print('time for full eigenval decomposition:', (end_e-start_e))
    #print("accuracy eig decomposition:", torch.dist( (U_X @ S_X @ torch.linalg.inv(U_X)), tensor_X))
    if torch.is_complex(U_X):
        U_X = U_X.real
        


    nV2_avg = (avg_X*torch.mean(tensor_Y)).to('cpu').numpy()
    if verbose:
        print("nV2_avg:", nV2_avg)

    eigenvals_Y, U_Y = torch.linalg.eigh(tensor_Y, UPLO='U') 
	
    U_Y = U_Y.real

    S_Y = torch.diag(eigenvals_Y.real)


	#print("dtype eigenvalues:", S_X.dtype, S_Y.dtype)
	#print("dtype eigenvectors:", U_X.dtype, U_Y.dtype)

    tensor_H = H_matrix(n, tensor_device, tensor_type=tensor_type) #from_dlpack(H.toDlpack())

    HUX = torch.mm(tensor_H, U_X)
    HUY = torch.mm(tensor_H, U_Y)	
    tUXH = torch.mm(torch.t(U_X), tensor_H)
    tUYH = torch.mm(torch.t(U_Y),tensor_H)


	#print(HUX.dtype, HUY.dtype, tUXH.dtype, tUYH.dtype)
    if go_gpu:
        torch.cuda.synchronize()
    #t_bm= time.time()
    nV2 = (torch.sum(torch.diag(torch.mm(torch.mm(torch.mm(torch.mm(HUX, S_X), torch.mm(tUXH, HUY )),  S_Y), tUYH )))).to('cpu').numpy()

    nV2 = 1.0*nV2/n


	#nV2Variance = cu.asnumpy(cu.sum(cu.diag(cu.matmul(cu.matmul(cu.matmul(cu.matmul(HUX, S_X) , (cu.matmul(tUXH, HUX ))), S_X), tUXH))) 
	#* cu.sum(cu.diag(cu.matmul(cu.matmul(cu.matmul(cu.matmul(HUY, S_Y), (cu.matmul(tUYH, HUY))), S_Y), tUYH)))
    nV2_variance = (torch.sum(torch.diag(torch.mm(torch.mm(torch.mm(torch.mm(HUX, S_X) , torch.mm(tUXH, HUX)), S_X), tUXH))) * torch.sum(torch.diag(torch.mm(torch.mm(torch.mm(torch.mm(HUY, S_Y), torch.mm(tUYH, HUY)), S_Y), tUYH)))).to('cpu').numpy()		
    
    nV2Variance = 2.0*(n-4)*(n-5)/n/(n-1)/(n-2)/(n-3) * nV2_variance / pow(n, 4.0)*pow(n, 2.0)

    del HUX, HUY, tUXH, tUYH, S_X, S_Y, U_X, U_Y, eigenvals_Y, tensor_Y, eigenvals_X,  tensor_X
    gc.collect()
    if go_gpu:    
        torch.cuda.empty_cache()
    
    if verbose:
        print('variance', nV2Variance)
    alpha = (nV2_avg*nV2_avg)/nV2Variance
    beta = nV2Variance/(nV2_avg)
    pvalue = 1.0-gamma.cdf(x=nV2, a=alpha, scale=beta)
	#print('pvalue',pvalue)
    return pvalue

