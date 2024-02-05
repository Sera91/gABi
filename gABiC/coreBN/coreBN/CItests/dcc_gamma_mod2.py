import numpy as np
from scipy.stats import gamma
from numba import njit



def H_matrix(n):
    H = np.eye(n,n) - (1/n)*np.ones((n,n))
    return H



@njit()
def Eucl_matrix_L1(x):
  n=len(x)
  out = np.zeros((n,n))
  for i in range(n):
      for j in range(n):
          out[i,j] = abs(x[i] - x[j])
  return out


	

def Dcov_gamma_py_cpu(x_arr, y_arr, index=1):
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
	n = len(x_arr)

	H = H_matrix(n) 
	if index==1:
	   mat_X = Eucl_matrix_L1(x_arr)
	   mat_Y = Eucl_matrix_L1(y_arr)
	e_values_X, e_vectors_X = np.linalg.eig(mat_X)#np.linalg.eigh(mat_X, UPLO='L')#pyspectra.eigsh(mat_X, 100) 
	e_values_Y, e_vectors_Y = np.linalg.eig(mat_Y)# np.linalg.eigh(mat_Y, UPLO='L')#pyspectra.eigsh(mat_Y, 100)  #np.linalg.eig(mat_X)

	#print(np.argsort(e_values_X))
        

	S_X = np.diag(e_values_X)
	print('sum Sx', np.sum(S_X))
	S_Y = np.diag(e_values_Y)
	U_X = e_vectors_X
	#print('Ux', U_X)
	U_Y = e_vectors_Y

  
	nV2 = np.sum(np.diag(np.matmul(np.matmul(np.matmul(np.matmul(np.dot(H,U_X), S_X), (np.matmul(np.dot(U_X.transpose(), H), np.dot(H, U_Y)))),  S_Y), np.dot(U_Y.transpose(),H)) ))/n
	print('nV2', nV2)
	nV2_avg = np.mean(mat_X)*np.mean(mat_Y)
	print('nV2_avg', nV2_avg)

	nV2Variance = 2.0*(n-4)*(n-5)/n/(n-1)/(n-2)/(n-3) * np.sum(np.diag(np.matmul(np.matmul(np.matmul(np.matmul(np.dot(H, U_X), S_X) , (np.matmul(np.dot(U_X.transpose(),H), np.dot(H,U_X)))), S_X), np.dot(U_X.transpose(),H))) * np.sum(np.diag(np.matmul(np.matmul(np.matmul(np.matmul(np.dot(H,U_Y), S_Y), (np.matmul(np.dot(U_Y.transpose(), H), np.dot(H, U_Y)))), S_Y), np.dot(U_Y.transpose(),H)))))/pow(n,4.0)*pow(n,2.0)

	print('variance', nV2Variance)
	alpha = pow(nV2_avg, 2.0)/nV2Variance
	beta = nV2Variance/(nV2_avg)
	pvalue = 1.0-gamma.cdf(x=nV2, a=alpha, scale=beta)
	print('pvalue',pvalue)
	return pvalue

