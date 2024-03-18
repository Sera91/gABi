from coreBN.utils import lpb4
import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial import distance_matrix


def matrix2(mat):
        if(mat.shape[0] == 1):
            return mat.T
        return mat

def dist(mat):
        """
        It calculates the distance between using distance_matrix() function and formats the output 
        as a single dimensional vector replicating the statement c(t(dist(mat))) in R.
        """
        dist = distance_matrix(mat,mat)
        dist = dist[np.triu_indices(dist.shape[0],1)] 
        return np.array(dist)

def expandgrid(*itrs):
        from itertools import product
        product = list(product(*itrs))
        return {'Var{}'.format(i+1): [x[i] for x in product] for i in range(len(itrs))}

def normalize(x):
        if(x.std(ddof=1) > 0):
            return ((x-x.mean())/x.std(ddof=1))
        else:
            return (x-x.mean())

def normalizeMat(mat):
        """
        It calculates the standard deviation norm for every column of a matrix (mat).
        """
        # if the number of rows is zero
        if(mat.shape[0] == 0):
            mat = mat.T
        mat = np.apply_along_axis(normalize, 0, mat)
        return mat

def colMeans(vec):
        vec = np.array(vec)
        return np.mean(vec, axis=0)


def random_fourier_features(x, w=None, b=None, num_f=25, sigma=None, seed=None):
        """
        It returns the randomized features set of the given random variable. 
        Instead of computing the full Kernel matrix (or processing k(x-Xi) for each point), 
        they randomly select a few features out of the space, and build f(x) based on those 
        few randomly selected features. This speeds up the kernel computations by several orders of magnitude.
        It delivers  similar accuracy to the full computation, and in some cases better. 

        It takes x (random variable), num_f (the number of features required), sigma (smooth parameter of RBF kernel) and 
        seed (for controlling random number generation) as inputs.

        Default value of num_f is taken to be 25.
        """

        # if x is a vector make it a (n,1)
        x = matrix2(x)
        r = x.shape[0]  # n => datapoints
        try:
            c = x.shape[1]    # D => dimension of variable
        except:
            c = 1
        if((sigma is None) or (sigma == 0)):
            sigma = 1

        if(w is None):
            # set the seed to seed
            if(seed is not None):
                import numpy.random as npr
                npr.seed(seed)

            # Generate normal(0,1) with (num_f*c) values
            # Shape w: (num_f, c)
            w = (1/sigma)*norm.rvs(size=(num_f*c))
            w = w.reshape(num_f, c, order='F')

            # set the seed to seed
            if(seed is not None):
                npr.seed(seed)

            # Create a row vector b of (1,r) with each value is in the range of 0 to 2*pi
            # Shape of b = (num_f,n)
            b = npr.uniform(size=num_f)
            b = np.repeat(2*np.pi*b[:, np.newaxis], r, axis=1)

        feat = np.sqrt(2)*((np.cos(w[:num_f, :c] @ x.T + b[:num_f, :])).T)
        return (feat, w, b)


    

def RIT(x, y, num_f2=5, seed=None,r=500, approx=None):
        """
        It returns a list containing the p-value (p-val) and test-statistic (Sta).
        Tests whether x and y are unconditionally independent using Randomized Independent Test method 
        using Random Fourier Features to improve the computation time and uses an approximation method LPB
        (Lindsay Pilla Basak method) for approximating the null distribution.
        LPB provides the calibration so that we can provide a “confidence threshold” 
        to differentiate between random correlation and correlation that was most likely structural.

        It takes as input a Random variable x, Random variable y, num_f2 (the number of features),
        sigma (smooth parameter of RBF kernel) and r (maximum number of datapoints considered for RFF)
        
        Default value of num_f2 is 5 and r is 500.
        """
        x = np.matrix(x).T
        y = np.matrix(y).T


        if(np.std(x) == 0 or np.std(y) == 0):
            return 1   # this is P value

        x = matrix2(x)
        y = matrix2(y)

        r = x.shape[0]
        if(r > 500):
            r1 = 500
        else:
            r1 = r

        x = normalizeMat(x).T
        y = normalizeMat(y).T

        (four_x, w, b) = random_fourier_features(
            x, num_f=num_f2, sigma=np.median(dist(x[:r1, ])), seed=seed)
        (four_y, w, b) = random_fourier_features(
            y, num_f=num_f2, sigma=np.median(dist(y[:r1, ])), seed=seed)
        
        f_x = normalizeMat(four_x)
        f_y = normalizeMat(four_y)

        Cxy = np.cov(f_x, f_y, rowvar=False)
        Cxy = Cxy[:num_f2, num_f2:]  # num_f2,num_f2
        Sta = r*np.sum(Cxy**2)


        res_x = f_x - np.repeat(np.matrix(colMeans(f_x))
                                [:, np.newaxis], r, axis=1)
        res_y = f_y - np.repeat(np.matrix(colMeans(f_x))
                                [:, np.newaxis], r, axis=1)

        d = expandgrid(
            np.arange(0, f_x.shape[1]), np.arange(0, f_y.shape[1]))
        
        res = np.array(res_x[:, np.array(d['Var2'])]) * \
            np.array(res_y[:, np.array(d['Var1'])])
        res = np.matrix(res)

        Cov = 1/r * ((res.T) @ res)
        w, v = np.linalg.eigh(Cov)
        w = np.flip(w)
        
        w = [i for i in w if i > 0]

        if(approx == "lpd4"):
            w1 = w
            p = 1 - lpb4(np.array(w1), Sta)
            if(p == None or np.isnan(p)):
                from .hbe import hbe
                p = 1 - hbe(w1, Sta)

        return (p, Sta)


def Sta_perm(r_x,r_y,r):
	Cxy=np.cov(r_x,r_y);
	Sta = r*np.sum(Cxy**2)
	return Sta

def RCIT(x, y, z=None, num_f=25, num_f2=5, seed=None,r=500, approx=None):
        """
        It returns a list containing the p-value (p-val) and statistic (Sta).
        It tests whether x and y are conditionally independent given z. RCoT calls RIT if z is empty. 
        This method utilizes Random Fourier Features over traditional kernel methods to scale linearly 
        with sample size and achieve the high accuracy and efficiency. 

        It takes as input x (Random variable), y (Random variable), z (Random variable), 
        num_f (the number of features for conditioning set), num_f2 (the number of features for unconditioning sets)
        sigma (smooth parameter of RBF kernel), seed (for controlling random number generation)
        and r (maximum number of datapoints considered for RFF)

        Default Value of num_f is 25, num_f2 is 5 which is observed to give consistent and most accurate results 
        and the default value of r is 500.
        """

        x = np.matrix(x).T
        y = np.matrix(y).T
        
        # Unconditional Testing
        if(len(z) == 0 or z == None):
            (p, Sta) = RIT(x, y, num_f2, seed,r)
            return (None, Sta, p)
        
        z = np.matrix(z).T

        x = matrix2(x)
        y = matrix2(y)
        z = matrix2(z)

        # Convert later to lamnda function
        z1 = []
        try:
            c = z.shape[1]
        except:
            c = 1

        for i in range(z.shape[1]):
            if(z[:,i].std()>0):
                z1.append(i)

        z = z[:,z1]

        z = matrix2(z)
        try:
            d = z.shape[1]    # D => dimension of variable
        except:
            d = 1
        # Unconditional Testing
        if(len(z) == 0 or z.any() == None):
            (p, Sta) = RIT(x, y, num_f2, seed,r)
            return (None, Sta, p)

        # Sta - test statistic -> s
        # if sd of x or sd of y == 0 then x and y are independent
        if (x.std() == 0 or y.std() == 0):
            # p=1 and Sta=0
            out = (1, 0)
            return(out)

        # make it explicit as maxData
        if (r >  x.shape[0]):
            r1 =  x.shape[0]
        else:
            r1 = r
        
        r = x.shape[0]
        # Normalize = making it as mean =0 and std= 1
        x = normalizeMat(x).T
        y = normalizeMat(y).T
        if(d == 1):
            z = normalizeMat(z).T
        else:
            z = normalizeMat(z)

        newy = np.vstack((y,z))

        (four_z, w, b) = random_fourier_features(
            z[:, :d], num_f=num_f, sigma=np.median(dist(z[:r1, :])), seed=seed)

        (four_x, w, b) = random_fourier_features(
            x, num_f=num_f2, sigma=np.median(dist(x[:r1, ])), seed=seed)

        (four_y, w, b) = random_fourier_features(
            new_y, num_f=num_f2, sigma=np.median(dist(y[:r1, ])), seed=seed)
        f_x = normalizeMat(four_x)
        f_y = normalizeMat(four_y)  # n,numf2
        f_z = normalizeMat(four_z)  # n,numf


        # Next few lines will be Equation2 from RCoT paper
        Cxy = np.cov(f_x, f_y, rowvar=False)  # 2*numf2,2*numf2

        Cxy = Cxy[:num_f2, num_f2:]  # num_f2,num_f2

        Cxy = np.round(Cxy, decimals=7)

        Czz = np.cov(f_z, rowvar=False)  # numf,numf

	#Cholesky inversion
        I = np.eye(num_f)
        L = cholesky((Czz + (np.eye(num_f) * 1e-10)), lower=True)
        L_inv = solve_triangular(L, I, lower=True)
        i_Czz = L_inv.T.dot(L_inv)  # Choleski inverse mat with shape (numf,numf)  

        Cxz = np.cov(f_x, f_z, rowvar=False)[:num_f2, num_f2:]  # numf2,numf

        Czy = np.cov(f_z, f_y, rowvar=False)[:num_f, num_f:]  # numf,numf2

        z_i_Czz = f_z @ i_Czz  # (n,numf) * (numf,numf)
        e_x_z = z_i_Czz @ Cxz.T  # n,numf
        e_y_z = z_i_Czz @ Czy

        # approximate null distributions

        res_x = f_x-e_x_z # residual of fourier after it removes the effect of conditioning  vars
        res_y = f_y-e_y_z
	
        if (approx == "perm"):
	   #using permutation test
           nperm=1000
           Cxy_z = np.cov(res_x, res_y, rowvar=False)
           Sta = r*np.sum(Cxy_z^2)
	   
           Stas=np.array([])
           for ps in range(nperm):
                i_sample = np.random.choice(range(r), size=r, replace=False)
                Sta_i = Sta_perm(res_x[i_sample],res_y,r)
                Stas=np.append(Stas,Sta_i)

           p = 1-(np.sum(np.where(Stas <= Sta))/len(Stas))
           return (None,Sta, p)

        matmul = (Cxz @ (i_Czz @ Czy))
        Cxy_z = Cxy-matmul  # less accurate for permutation testing

        Sta = r * np.sum(Cxy_z**2)

        d = expandgrid(
            np.arange(0, f_x.shape[1]), np.arange(0, f_y.shape[1]))
        res = np.array(res_x[:, np.array(d['Var2'])]) * \
            np.array(res_y[:, np.array(d['Var1'])])
        res = np.matrix(res)
        
        Cov = 1.0/r * ((res.T) @ res)

        w, v = np.linalg.eigh(Cov) #this return eigenvalues, eigenvectors
        w = np.flip(w)

        w = [i for i in w if i > 0]

        if(approx == "lpd4"):
            # from lpb4 import lpb4
            w1 = w
            p = 1 - lpb4(np.array(w1), Sta)
            if(p == None or np.isnan(p)):
                from coreBN.utils import hbe
                p = 1 - hbe(w1, Sta)
        return (Cxy_z, Sta, p)
    
def independence(x, y, z=None, num_f=25, num_f2=5, seed=None,r=500, alpha=0.05):
        (Cxy,Sta,p) = rcit(x, y, z, num_f, num_f2, seed,r)
        dependence =  max(0, (.5 + (alpha-p)/(alpha*2)), (.5 - (p-alpha)/(2*(1-alpha))))
        return (1-dependence)

def dependence( x, y, z=None, num_f=25, num_f2=5, seed=None,r=500):
        independence = independence(x, y, z, num_f, num_f2, seed,r)
        return 1-independence
