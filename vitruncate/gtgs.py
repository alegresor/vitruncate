from qmcpy import Sobol
from numpy import *
from numpy.linalg import inv,det
from scipy.spatial.distance import pdist,squareform
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
    """
    Gaussian Truncated Distribution Generator by Stein Method.
    Code adapted from: https://github.com/DartML/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py
    """
    def __init__(self, n, d, mu, Sigma, L, U, epsilon, alpha=.9, seed=None):
        """
        Args:
            n (int): number of samples
            d (int): dimension
            mu (ndarray): length d vector of means
            Sigma (ndarray): d x d symmetric positive definite covariance matrix
            L (ndarray): length d vector of lower bounds
            U (ndarray): length d vector of upper bounds
            epsilon (ndarray): step size
            alpha (float): momentum of adaptive gradient
            seed (int): seed for reproducibility
        """
        self.n = n
        self.d = d
        self.mu = mu.reshape((-1,1))
        self.Sigma_norm = det(Sigma) # normalization factor
        self.Sigma = Sigma/self.Sigma_norm
        self.invSigma = inv(self.Sigma)
        self.B = (vstack((L.flatten(),U.flatten()))-self.mu.T)/sqrt(self.Sigma_norm) # 2 x d array of bounds
        self.epsilon = epsilon
        self.alpha = alpha
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
        self.fudge = 1e-6
        self.iter = 0
        self.hgrad = zeros((self.n,self.d),dtype=float)
    def _dlogpgt(self, x):
        maxd = 1e10
        below = x < self.B[0,:]
        above = x > self.B[1,:]
        inbounds = (~below)*(~above)
        t = -(x@self.invSigma)*inbounds + below*maxd - above*maxd
        return t
    def _k_rbf(self, x):
        pairwise_dists = squareform(pdist(x))**2
        h = median(pairwise_dists)  
        h = sqrt(.5*h/log(self.n+1))
        Kxy = exp(-pairwise_dists/(2*h**2))
        dxkxy = -Kxy@x
        sumkxy = Kxy.sum(1)
        dxkxy += x*sumkxy[:,None]
        dxkxy = dxkxy/(h**2)
        return Kxy,dxkxy 
    def walk(self, steps):
        for i in range(steps):
            lnpgrad = self._dlogpgt(self.x)
            kxy,dxkxy = self._k_rbf(self.x)  
            grad = ((kxy@lnpgrad)+dxkxy)/self.n  
            if self.iter==0:
                self.hgrad = self.hgrad+grad**2
            else:
                self.hgrad = self.alpha*self.hgrad+(1-self.alpha)*(grad**2)
            adj_grad = grad/(self.fudge+sqrt(self.hgrad))
            self.x += self.epsilon*adj_grad 
            self.iter += 1
        return self.get_val()
    def get_val(self):
        return self.x*sqrt(self.Sigma_norm)+self.mu.T