import numpy as np
from math import comb
from .utils_RCIT import *

def vapply(X,fun,v):
    """
    Vectorises the array x with the function fun.
    Any other arguments required for the function fun is given taken as input after x and fun.
    """
    res = []
    for i in range(X.shape[0]):
        res.append(fun(int(X[i]),v))
    res = np.array(res)
    return res

def mapply(fun,ivec,jvec,vec1,vec2):
    """
    Returns a single vector after vectorizing 2 arrays ivec and jvec with function fun.
    Vec1 and vec2 are  vectors that are required arguments for the function fun.
    """
    res=[]
    for i in ivec:
        l = []
        for j in jvec:
            l.append(fun(int(i),int(j),vec1,vec2))
        res.append(l)
    return res

def rep(x,y):
    return np.repeat(x,y)

def choose(n,kvec):
    ans = []
    for k in kvec:
        ans.append(comb(n,k))
    return np.array(ans)

def c(x, vec):
    return np.insert(vec,0,x)

def polyroot(vec):
    return np.polynomial.polynomial.polyroots(vec)

def Re(vec):
    res = [n for n in vec if np.isreal(n)]
    return res

def uniroot(fun, limit, m_vec, N, tol):
    from scipy.optimize import brentq as root
    x = root(fun,limit[0],limit[1], args=(m_vec, N), xtol=tol)
    return x

def factorial(vec):
    ans = []
    for x in vec:
        ans.append(np.math.factorial(x))
    return np.array(ans)

def lpb4(coeff, x):
    """
    Computes the cdf (Cummalitive Distribution Function) of a positively-weighted sum of 
    chi-squared random variables with the Lindsay-Pilla-Basak method using four support points.
    LPB approximation technique is used by RCoT to determine if the answer is statistically significant or 
    it could have been generate by a random variation.
    
    In some cases when the coefficient vector was of length two or three, the algorithm would be 
    unable to find roots of a particular equation during an intermediate step, and so 
    the algorithm would produce NULL solutions. If the coefficient vector is of length less than four, 
    the Hall-Buckley-Eagleson (HBE) method is used.

    It takes as input the coeff (coefficient vector of atleast length of 4) and x (a vector).
    The values of the coefficient vector and vector (x) should be positive.
    """
    
    if(coeff.any() == None or x == None):
        print("missing an argument - need to specify \"coeff\" and \"x\"")
        return None

    if (checkCoeffsArePositiveError(coeff)):
        exit(getCoeffError(coeff))

    if (checkXvaluesArePositiveError([x])):
        exit(getXvaluesError([x]))

    if(len(coeff) < 4):
        print(
            "Less than four coefficients - LPB4 method may return NaN: running hbe instead.")
        from hbe import hbe
        return(hbe(coeff, x))

    # step 0: decide on parameters for distribution and support points p specified to be 4 for this version of the function
    p = 4

    # step 1: Determine/compute the moments m_1(H), ... m_2p(H)
    moment_vec = get_weighted_sum_of_chi_squared_moments(coeff, p)

    lambdatilde_1 = get_lambdatilde_1(moment_vec[0], moment_vec[1])

    bisect_tol = 1e-9
    lambdatilde_p = get_lambdatilde_p(lambdatilde_1, p, moment_vec, bisect_tol)

    M_p = deltaNmat_applied(lambdatilde_p, moment_vec, p)

    mu_poly_coeff_vec = get_Stilde_polynomial_coefficients(np.array(M_p))

    roots = np.polynomial.polynomial.polyroots(mu_poly_coeff_vec)
    mu_roots = np.float64(Re(np.complex128(roots)))

    pi_vec = generate_and_solve_VDM_system(np.array(M_p), mu_roots)

    mixed_p_val_vec = get_mixed_p_val_vec(x, mu_roots, pi_vec, lambdatilde_p)

    return mixed_p_val_vec
 

def get_weighted_sum_of_chi_squared_moments(coeffvec, p):
    # Checked - giving correct
    cumulant_vec = get_cumulant_vec_vectorised(coeffvec, p)
    moment_vec = get_moments_from_cumulants(cumulant_vec)
    return (moment_vec)

def get_cumulant_vec_vectorised(coeffvec, p):
    index = np.arange(1,(2*p)+1)

    cumulant_vec = (2**(index-1)) * factorial(index-1) * vapply(index, sum_of_powers, coeffvec)

    return cumulant_vec

def sum_of_powers(index, v):
    return np.sum(v**index)

def get_moments_from_cumulants(cumulant_vec):
    moment_vec = np.copy(cumulant_vec)
    if(len(moment_vec) > 1):
        for n in range(1,len(moment_vec)):
            moment_vec[n] = moment_vec[n] + update_moment_from_lower_moments_and_cumulants(n+1, moment_vec, cumulant_vec)
    return moment_vec

def update_moment_from_lower_moments_and_cumulants(n, moment_vec, cumulant_vec):
    m = np.arange(1,n)
    sum_of_additional_terms = np.sum(choose(n-1, m-1) * cumulant_vec[m-1] * moment_vec[n-m-1])
    return sum_of_additional_terms

def get_lambdatilde_1(m1, m2):
    return (m2/(m1**2)-1)

def deltaNmat_applied(x, m_vec, N):
    Nplus1 = N+1
    #want moments 0, 1, ..., 2N
    m_vec = m_vec[:(2*N)]
    m_vec = np.insert(m_vec,0,1)
    #these will be the coefficients for the x in (1+c_1*x)*(1+c_2*x)*...
	#want coefficients 0, 0, 1, 2, .., 2N-1 - so 2N+1 in total 
    coeff_vec = np.arange(0,2*N)
    coeff_vec = (np.insert(coeff_vec, 0, 0))*x + 1
    #not necessary to initialise, could use length(m_vec) below, but we do it anyway for readability
    prod_x_terms_vec = np.repeat(0, 2*N+1)
    prod_x_terms_vec = 1/vapply(np.arange(1,len(prod_x_terms_vec)+1), get_partial_products, coeff_vec)
    i_vec = np.arange(0,Nplus1)
    j_vec = np.arange(0,Nplus1)
    delta_mat = mapply(get_index_element, i_vec, j_vec, m_vec, prod_x_terms_vec)
    return delta_mat

def get_partial_products(index, vec):
    return np.prod(vec[:index])

def get_index_element(i,j,vec1, vec2):
    index = i+j
    return (vec1[index] * vec2[index])

def det_deltamat_n(x, m_vec, N):
    res = np.linalg.det(deltaNmat_applied(x, m_vec, N))
    return res

def get_lambdatilde_p(lambdatilde_1, p, moment_vec, bisect_tol):
    lambdatilde_vec = rep(0.0, p)
    lambdatilde_vec[0] = lambdatilde_1
    bisect_tol = 1e-9

    if(p>1):
        for i in range(1,p):
            lambdatilde_vec[i] = uniroot(det_deltamat_n, [0, lambdatilde_vec[i-1]], m_vec=moment_vec, N=i+1, tol=bisect_tol)
    lambdatilde_p = lambdatilde_vec[p-1]
    return lambdatilde_p

    
def get_Stilde_polynomial_coefficients(M_p):
    n = M_p.shape[0]
    index = np.arange(1,n+1)
    mu_poly_coeff_vec = vapply(index, get_ith_coeff_of_Stilde_poly, M_p)

    return mu_poly_coeff_vec

def get_base_vector(n,i):
    base_vec = rep(0, n)
    base_vec[i] = 1
    return base_vec

def get_ith_coeff_of_Stilde_poly(i, mat):
    n = mat.shape[0]
    base_vec = get_base_vector(n,i-1)
    mat[:,n-1] = base_vec
    return (np.linalg.det(mat))

def generate_and_solve_VDM_system(M_p, mu_roots):
    b_vec = get_VDM_b_vec(M_p)
    VDM = generate_van_der_monde(mu_roots)
    pi_vec = np.linalg.solve(VDM, b_vec)
    return pi_vec

#simply takes the last column, and removes last element of last column
def get_VDM_b_vec(mat):
    b_vec = mat[:,0]
    b_vec = b_vec[:-1]
    return b_vec

#generates the van der monde matrix from a vector
def generate_van_der_monde(vec):
    p = len(vec)
    vdm = np.zeros((p,p))
    for i in range(p):
        vdm[i,:] = (vec**i)
    return vdm

def get_mixed_p_val_vec(quantile_vec, mu_vec, pi_vec, lambdatilde_p):
    p = len(mu_vec)
    alpha = 1/lambdatilde_p
    beta_vec = mu_vec/alpha
    try:
        l = len(quantile_vec)
    except:
        l = 1
    partial_pval_vec = rep(0, l)
    
    for i in range(0,p):
        partial_pval_vec = partial_pval_vec + pi_vec[i] * pgamma(quantile_vec, shape=alpha, rate = beta_vec[i])
        
    return partial_pval_vec

def compute_composite_pgamma(index, qval, shape_val, scale_vec):
    return pgamma(qval, shape=shape_val, rate = scale_vec[index])
