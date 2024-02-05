import numpy as np
from math import factorial
from itertools import combinations
from scipy.stats import gamma



def rSubset(n, r):
    # return list of all subsets of length r
    # to deal with duplicate subsets use
    # set(list(combinations(arr, r)))
    return list(combinations(np.arange(n), r))


def distance_sq_matrix(x_arr, N_obs):
    tile_matrix = np.tile(x_arr, (N_obs, 1))
    return np.power(np.abs(tile_matrix - tile_matrix.T), 2.0)


def kernel_matrix(x_arr, sig=1.0):
    N_obs = len(x_arr)
    dist_K = distance_sq_matrix(x_arr, N_obs)
    K = np.exp(-dist_K/(2.0*sig**2))
    return K


def H_matrix(n):
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    return H


def coeff_tuple(n, k):
    if k < n:
        return factorial(n)/factorial(n-k)
    else:
        print("Error: k should be less than n")
        return None


def Mu_mat(K):
    n = K.shape[0]
    sel_indices = rSubset(n, 2)
    sum_K = 0.0
    for index in sel_indices:
        # print(index)
        sum_K = sum_K + K[index]
    denom = n*(n-1)
    return sum_K/denom


def trace_matrix(M):
    return np.diagonal(M).sum()


# function returning the expected value of the HSIC biased empirical estimate
def E_val_HSIC(K, L, sig=1.0):
    N_obs = len(x_arr)
    mu_X = Mu_mat(K)
    mu_Y = Mu_mat(L)
    result = (1/N_obs)*(1.0 - mu_X)*(1.0-mu_Y)
    return result


def Var_val_HSIC(K, L, m):
    denom = m*(m-1)*(m-2)*(m-3)  # coeff_tuple(m,4)
    num = 2*(m-4)*(m-5)
    H = H_matrix(n)
    K_bar = H.T @ (K @ H)
    L_bar = H.T @ (L @ H)
    K_prod = K @ (K_bar)
    L_prod = L @ (L_bar)
    C_xx = (1/m**2)*trace_matrix(K_prod)
    C_yy = (1/m**2)*trace_matrix(L_prod)

    return (num/denom)*C_xx*C_yy


def Hsic_gamma(x, y, sigma=1):
    # ' @details Let x and y be two samples of length n.
    # Gram matrices K and L are defined as: K_{i,j} =exp((x_i-x_j)^2/sig^2)   (\eqn{K_{i,j} = \exp\frac{(x_i-x_j)^2}{\sigma^2}})
    #                                      L_{i,j} =exp((y_i-y_j)^2/sig^2)}.
    # Defining H Matrix with elements : \eqn{H_{i,j} = \delta_{i,j} - \frac{1}{n}.
    # Let \eqn{A=HKH} and \eqn{B=HLH}, then \eqn{HSIC(x,y)=\frac{1}{n^2}Tr(AB)}.
    # Gamma test compares HSIC(x,y) with the alpha quantile of the gamma distribution with mean and variance such as HSIC under independence hypothesis.

    N_obs = len(x)
    K = kernel_matrix(x, sig=sigma)
    L = kernel_matrix(y, sig=sigma)
    H = H_matrix(N_obs)
    hsic = (1/pow(N_obs, 2))*trace_matrix(K @ (H @ (L @ H)))
    E_HSIC = E_val_HSIC(K, L, 1)
    Var_HSIC = Var_val_HSIC(K, L, m)
    alpha = pow(E_HSIC, 2.0) / Var_HSIC
    beta = Var_HSIC/E_HSIC

    p_value = 1 - gamma.ppf(q=hsic, a=alpha, scale=1.0/beta)
    return p_value
