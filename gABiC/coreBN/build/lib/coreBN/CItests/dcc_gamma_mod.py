import numpy as np
from scipy.stats import gamma
from numba import njit



def H_matrix(n):
    H = np.eye(n,n) - (1.0/n)*np.ones((n,n))
    return H



@njit()
def Eucl_matrix_L1(x):
  n=len(x)
  out = np.zeros((n,n), dtype=np.float64)
  for i in range(n):
      for j in range(n):
          out[i,j] = abs(x[i] - x[j])
  return out


@njit()
def Eucl_matrix_L1_fast(x,n):
	out = np.zeros((n,n))
	for i in range(n):
		for j in range(i+1,n):
			out[i,j] = abs(x[i] - x[j])
			out[j,i] = out[i,j]
	return out


	

def Dcov_gamma_py(x_arr, y_arr, index=1):
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
 
	
	mat_X = np.real(Eucl_matrix_L1_fast(x_arr, n))
	mat_Y = np.real(Eucl_matrix_L1_fast(y_arr, n))

	nV2_avg = np.mean(mat_X)*np.mean(mat_Y)

	e_values_X, e_vectors_X = np.linalg.eig(mat_X)#np.linalg.eigh(mat_X, UPLO='L')#pyspectra.eigsh(mat_X, 100) 
	e_values_Y, e_vectors_Y = np.linalg.eig(mat_Y)# np.linalg.eigh(mat_Y, UPLO='L')#pyspectra.eigsh(mat_Y, 100)  #np.linalg.eig(mat_X)

	#print(np.argsort(e_values_X)) 

	S_X = np.diag(np.real(e_values_X))
	#print('sum Sx', np.sum(S_X))
	S_Y = np.diag(np.real(e_values_Y))
	U_X = np.real(e_vectors_X)
	#print('Ux', U_X)
	U_Y = np.real(e_vectors_Y)

	H = H_matrix(n)

	nV2 = np.sum(np.diag(np.dot(H,U_X) @ S_X @ (np.dot(U_X.transpose(), H) @ np.dot(H, U_Y)) @ S_Y @ np.dot(U_Y.transpose(),H) ))/n
	#print('nV2:', nV2, 'nV2_avg:', nV2_avg)

	nV2Variance = 2.0*(n-4)*(n-5)/n/(n-1)/(n-2)/(n-3) * np.sum(np.diag(np.dot(H, U_X) @ S_X @ (np.dot(U_X.transpose(),H)@(np.dot(H,U_X)) @ S_X @ np.dot(U_X.transpose(),H))) * np.sum(np.diag( np.dot(H,U_Y) @ S_Y @ (np.dot(U_Y.transpose(), H) @ np.dot(H, U_Y)) @ S_Y @ np.dot(U_Y.transpose(),H))))/pow(n,4.0)*pow(n,2.0)

	#print('variance', nV2Variance)
	alpha = pow(nV2_avg, 2.0)/nV2Variance
	beta = nV2Variance/(nV2_avg)

	#print('alpha',alpha)
	#print('beta', beta)

	pvalue = 1.0-gamma.cdf(x=nV2, a=alpha, scale=beta)
	print('pvalue',pvalue)


	return pvalue
