import numpy as np
from scipy.stats import gamma
import cupy as cu



def H_matrix(n):
    
    H = cu.eye(n,n) - (1/n)*cu.ones((n,n))
	
    return H


def Eucl_matrix_L1(x):
  n=len(x)
  out = cu.zeros((n,n))
  for i in range(n):
      for j in range(n):
          out[i,j] = abs(x[i] - x[j])
  return out


	

def Dcov_gamma_py_gpu(x_arr, y_arr, index=1):

	if (len(x_arr) != len(y_arr)):
		print("ERROR: the sample size of arrays is different!!")
		return 0
	
	if (index < 0 or index > 2):
		print("ERROR: index must be in the interval [0,2)!")
		return 0
	
	if not(np.isfinite(x_arr).all()): 
		print("Data contains missing or infinite values")
		return 0
	
	if not(np.isfinite(y_arr).all()): 
		print("Data contains missing or infinite values")
		return 0

	n = len(x_arr) #number of observations/rows in dataframe

	#if GPU_work:
	g_xarr= cu.asarray(x_arr)
	g_yarr= cu.asarray(y_arr)	
 
	mat_X = Eucl_matrix_L1(g_xarr)
	mat_Y = Eucl_matrix_L1(g_yarr)

	del g_xarr
	del g_yarr

	nV2_avg =cu.asnumpy(cu.mean(mat_X)*cu.mean(mat_Y))
	
	e_values_X, e_vectors_X = cu.linalg.eigh(mat_X)#np.linalg.eigh(mat_X, UPLO='L')#pyspectra.eigsh(mat_X, 100) 
	
	del mat_X

	e_values_Y, e_vectors_Y = cu.linalg.eigh(mat_Y)# np.linalg.eigh(mat_Y, UPLO='L')#pyspectra.eigsh(mat_Y, 100)  #np.linalg.eig(mat_X)

	#print(np.argsort(e_values_X))
	del mat_Y

	H = H_matrix(n)

	S_X = cu.diag(e_values_X)
	#print('sum Sx', cu.sum(S_X))
	S_Y = cu.diag(e_values_Y)
	U_X = e_vectors_X
	#print('Ux', U_X)
	U_Y = e_vectors_Y

	del e_values_X, e_values_Y

	nV2 = cu.asnumpy(cu.sum(cu.diag(cu.matmul(cu.matmul(cu.matmul(cu.matmul(cu.dot(H,U_X), S_X), (cu.matmul(cu.dot(U_X.transpose(), H), cu.dot(H, U_Y)))),  S_Y), cu.dot(U_Y.transpose(),H)) )))/n  
	#print('nV2', nV2)
	
	#print('nV2_avg', nV2_avg)

	nV2Variance = 2.0*(n-4)*(n-5)/n/(n-1)/(n-2)/(n-3) *cu.asnumpy(cu.sum(cu.diag(cu.matmul(cu.matmul(cu.matmul(cu.matmul(cu.dot(H, U_X), S_X) , (cu.matmul(cu.dot(U_X.transpose(),H), cu.dot(H,U_X)))), S_X), cu.dot(U_X.transpose(),H))) * cu.sum(cu.diag(cu.matmul(cu.matmul(cu.matmul(cu.matmul(cu.dot(H,U_Y), S_Y), (cu.matmul(cu.dot(U_Y.transpose(), H), cu.dot(H, U_Y)))), S_Y), cu.dot(U_Y.transpose(),H))))))/pow(n,4.0)*pow(n,2.0)

	del S_X, S_Y, U_X, U_Y, H

	#print('variance', nV2Variance)
	alpha = pow(nV2_avg, 2.0)/nV2Variance

	beta = nV2Variance/(nV2_avg)
	
	pvalue = 1.0-gamma.cdf(x=nV2, a=alpha, scale=beta)
	#print('pvalue',pvalue)
	return pvalue

