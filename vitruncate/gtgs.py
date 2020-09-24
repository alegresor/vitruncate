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
    def _k_rbf(self, x):
        pairwise_dists = squareform(pdist(x))**2
        h = sqrt(.5*median(pairwise_dists)/log(self.n+1))
        Kxy = exp(-pairwise_dists/(2*h**2))
        dxkxy = -1*(Kxy@x)
        sumkxy = Kxy.sum(1)
        for i in range(self.d):
            dxkxy[:, i] = dxkxy[:,i]+(x[:,i]@sumkxy)
        dxkxy = dxkxy/(h**2)
        return Kxy,dxkxy
    def _dlogpgt(self, x):
        valid = (x>self.B[0,:]).all(1)&(x<self.B[1,:]).all(1)
        t = -(x@self.invSigma)#*valid[:,None]
        return t
    def _phiHatStar(self, x):
        lnpgrad = self._dlogpgt(self.x)
        kxy,dxkxy = self._k_rbf(self.x)  
        grad = ((kxy@lnpgrad)+dxkxy)/self.n
        return grad
    '''
    def _k_rbf(self, x, z): 
        h = 5
        k = zeros((self.n,self.n),dtype=float)
        dk = zeros((self.n,self.n),dtype=float)
        for i in range(self.n):
            for j in range(i):
                dist = z[j,:]-x[i,:]
                k[i,j] = exp((dist**2).sum()/h)
                dk[i,j] = 2*k[i,j]*dist.sum()/h
        k += k.T + diag(ones(self.n)) # symmetric with ones along the diagnol
        dk -= dk.T # skew matrix with zeros along the diagnol
        return k,dk
    def _dlnpgt(self, x):
        valid = (x>self.B[0,:]).all(1)&(x<self.B[1,:]).all(1)
        t = - x@inv(self.Sigma)*valid[:,None]
        return t
    def _phiHatStar(self, z):
        k,dk = self._k_rbf(self.x,z)
        dlnp = self._dlnpgt(self.x)
        t = zeros((self.n,self.d),dtype=float)
        for i in range(self.n):
            t[i,:] = (k[:,i,None]*dlnp+dk[:,i,None]).mean(0)
        return t
    '''
    def _step(self):
        grad = self._phiHatStar(self.x)
        if self.iter==0:
            self.hgrad += grad**2
        else:
            self.hgrad = self.alpha*self.hgrad+(1-self.alpha)*(grad**2)
        adj_grad = grad/(self.fudge+sqrt(self.hgrad))
        self.x += self.epsilon*adj_grad 
        self.iter += 1
    def walk(self, steps):
        for i in range(steps):
            self._step()
        return self.get_val()
    def get_val(self):
        return self.x*sqrt(self.Sigma_norm)+self.mu.T