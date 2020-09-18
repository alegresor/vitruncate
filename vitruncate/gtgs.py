from qmcpy import Sobol
from numpy import *
from numpy.linalg import inv,det
from scipy import stats

def trunc_gen(x_stdu, lb, ub, distrib, independent=True, **params):
    """
    Transform samples mimicking a standard uniform to mimic a truncated distribution
    
    Args:
        x_stdu (numpy.ndarray): n samples x d dimensional array of samples
        lb (numpy.ndarray): length d lower bound vector
        ub (numpy.ndarray): length d upper bound vector
        distrib (scipy.stats module): a distribution from scipy.stats (i.e.norm,t) 
            that has `cdf` and `ppf` (inverse cdf) functions
        independent (bool): sample dimensions are independent? 
        **params (dict): keyword arguments (parameters) to pass to into `distrib.cdf` and `distrib.ppf`
            
    Return:
        numpy.ndarray: n samples by d dimension array of samples mimicking the truncated distribution
    """
    cdf = distrib.cdf
    invcdf = distrib.ppf
    if independent:
        x_trunc = zeros(x_stdu.shape,dtype=double)
        for j in range(x_trunc.shape[1]):
            params_j = {param:val[j] for param,val in params.items()}
            cdflb = cdf(lb[j],**params_j)
            cdfub = cdf(ub[j],**params_j)
            x_trunc[:,j] = invcdf((cdfub-cdflb)*x_stdu[:,j]+cdflb,**params_j)
        return x_trunc
    else:
        msg = '''
        `trunc_gen` currenly only supports generating samples from 
        distributions with independent dimensions.
        '''
        raise Exception(msg)
        return -1
        
class GTGS(object):
    """ Gaussian Truncated Distribution Generator by Stein Method"""
    def __init__(self, n, d, mu, Sigma, L, U, epsilon, seed=None):
        """
        Args:
            n (int): number of samples
            d (int): dimension
            mu (ndarray): length d vector of means
            Sigma (ndarray): d x d symmetric positive definite covariance matrix
            L (ndarray): length d vector of lower bounds
            U (ndarray): length d vector of upper bounds
            epsilon (ndarray): step size
            seed (int): seed for reproducibility
        """
        self.n = n
        self.d = d
        self.mu = mu.reshape((-1,1))
        self.Sigma_norm = det(Sigma) # normalization factor
        self.Sigma = Sigma/self.Sigma_norm
        self.B = (vstack((L.flatten(),U.flatten()))-self.mu.T)/sqrt(self.Sigma_norm) # 2 x d array of bounds
        self.epsilon = epsilon
        ss = Sobol(d,seed=seed).gen_samples(self.n) # Sobol' samples from QMCPy. Could be replaced with IID samples
        x_init = trunc_gen(
            x_stdu = ss,
            lb = self.B[0,:],
            ub = self.B[1,:],
            distrib = stats.norm,
            independent = True,
            loc = zeros(self.d),
            scale = sqrt(self.Sigma.diagonal()))
        self.x_init = x_init
        self.x = x_init
        self.h = 10
    def k_rbf(self, x, z): 
        """
        RBF Kernel
        
        Args:
            x (ndarray): n x d array of current samples
            z (ndarray): n x d array of update samples
        
        Return: 
            ndarray: n x n array of samples.
                - ndarray_{i,j} = k(x_i,z_j). 
                - For RBF Kernel the ndarray is symmetric
                - For RBF Kernel the diagnol is 1.
        """
        t = zeros((self.n,self.n),dtype=float)
        for i in range(self.n):
            for j in range(i):
                t[i,j] = exp(((z[j,:]-x[i,:])**2).sum()/self.h)
        t += t.T
        t += ones(self.n)
        return t
    def dk_rbf(self, x, z):
        """
        Derivitive of RBF Kernel
        
        Args:
            x (ndarray): current samples
            z (ndarray): n x d array of update samples
        Return: 
            ndarray: n x n array of samples.
                - ndarray_{i,j} = d(k(x_i,z_j)) / (d x_i). 
                - For RBF Kernel the ndarray is a skew symmetric
                - For RBF Kernel the diagnol is 0.
        """
        t = zeros((self.n,self.n),dtype=float)
        for i in range(self.n):
            for j in range(i):
                t[i,j] = 2*self.k_rbf_curr[i,j]*(z[j,:]-x[i,:]).sum()/self.h
        t -= t.T
        return t
    def dlogpgt(self, x):
        """
        Derivitive of log of probability for truncated Gaussian
        
        Args:
            x (ndarray): n x d array of current samples
        
        Return:
            ndarray: n x d array of derivitives
        """
        valid = (x>self.B[0,:]).all(1)&(x<self.B[1,:]).all(1)
        t = - x@inv(self.Sigma)*valid[:,None]
        return t
    def phiHatStar(self, z):
        """
        Get the update direction.
        
        Args:
            z (ndarray): n x d array of samples to update.
        
        Return:
            ndarray: n x d array of update directions.
        """
        self.k_rbf_curr = self.k_rbf(self.x,z)
        self.dlogpgt_curr = self.dlogpgt(self.x)
        self.dk_rbf_curr = self.dk_rbf(self.x,z)
        t = zeros((self.n,self.d),dtype=float)
        for i in range(self.n):
            t[i,:] = (self.k_rbf_curr[:,i,None]*self.dlogpgt_curr+self.dk_rbf_curr[:,i,None]).mean(0)
        return t
    def get_curr_x(self):
        """
        Get the current samples.
        
        Return: 
            ndarray: n x d array of current samples.
        """
        return self.x*sqrt(self.Sigma_norm)+self.mu.T
    def step(self):
        """
        Step the samples in the update direction. 
        
        Return:
            ndarray: current samples.
        """
        phs = self.phiHatStar(self.x)
        self.x += self.epsilon*phs
        return self.get_curr_x()
    def walk(self, steps):
        """
        Take multiple update steps. 
        
        Args:
            steps (int): number of times to call step. 
        
        Return: 
            ndarray: current samples. 
        """
        for i in range(steps):
            self.step()
        return self.get_curr_x()