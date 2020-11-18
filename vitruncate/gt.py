from numpy import *
from numpy.linalg import *
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal as mvn, norm
from math import comb
from qmcpy import *

        
class GT(object):
    """
    Gaussian Truncated Distribution Generator by Stein Method.
    Code adapted from: https://github.com/DartML/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py
    """
    def __init__(self, n, d, mu, Sigma, L, U, init_type='IID', seed=None, n_block=None, alpha_r=1e-5):
        """
        Args:
            n (int): number of samples
            d (int): dimension
            mu (ndarray): length d vector of means
            Sigma (ndarray): d x d symmetric positive definite covariance matrix
            L (ndarray): length d vector of lower bounds
            U (ndarray): length d vector of upper bounds
            init_type (str): "Sobol" or "IID" point initialization size
            seed (int): seed for reproducibility
            n_block (int): number of samples in a computation block.
            alpha_r (float): percentage of mass acceptable to discard when forcing finite bounds.
        """
        self.n = n
        self.d = d
        self.mu = array(mu,dtype=float).flatten()
        self.Sigma = array(Sigma,dtype=float)
        self.L = array(L).flatten()
        self.U = array(U).flatten()
        self.itype = init_type.upper()
        self.seed = seed
        self.n_block = n_block if n_block else self.n
        self.alpha_r = alpha_r 
        self.L_hat,self.U_hat = self._rebound()
        self.mut = self.mu - self.mu
        self.independent = (self.Sigma==(self.Sigma*eye(self.d))).all()
        W = diag(1/(self.U_hat-self.L_hat))
        self.St = W@self.Sigma@W.T 
        self.invSt = inv(self.St)
        self.Lt = (self.L_hat-self.mu)@W
        self.Ut = (self.U_hat-self.mu)@W
        if self.n_block != self.n:
            raise Exception("n_block not implemented yet, please default.")
        self.blocks = int(floor(self.n/self.n_block))
        if self.n<2 or self.n_block<2 or self.d<1:
            raise Exception("n and n_block must be >=2 and d must be >0.")
        self.x_stdu = self._get_stdu_pts(self.n)  
        self.x_init = zeros((self.n,self.d),dtype=float)
        std = sqrt(self.St.diagonal())
        cdflb = norm.cdf(self.Lt,scale=std)
        cdfub = norm.cdf(self.Ut,scale=std)
        self.x_init = norm.ppf((cdfub-cdflb)*self.x_stdu+cdflb,scale=std)
        self.fudge = 1e-6
        self.reset()
        alpha = 1e-5 # expect 1 in every 100k out of bounds
        self.g_hat_l,self.g_hat_u = self._approx_mass_qmc(self.L_hat,self.U_hat)
        coefs = array([alpha*self.g_hat_l/(1-alpha)] + [-comb(d,j)*2**j for j in range(1,self.d+1)], dtype=double)
        self.beta = real(roots(coefs).max())+1
    def _rebound(self):
        if isfinite(self.L).all() and isfinite(self.U).all():
            return self.L,self.U
        eps = finfo(float).eps
        L_hat = zeros(self.d)
        U_hat = zeros(self.d)
        for j in range(self.d):
            L_hat[j] = norm.ppf(eps,loc=self.mu[j],scale=sqrt(self.Sigma[j,j])) if self.L[j]==(-inf) else self.L[j]
            U_hat[j] = norm.ppf(1-eps,loc=self.mu[j],scale=sqrt(self.Sigma[j,j])) if self.U[j]==inf else self.U[j]
        return L_hat,U_hat
    def _approx_mass_qmc(self,L,U):
        g = Gaussian(Sobol(self.d), self.mu, self.Sigma)
        gpdf = CustomFun(g,lambda x: ((x>=L).all(1)*(x<=U).all(1)).astype(float))
        mass,data = CubQMCSobolG(gpdf, abs_tol=1e-3).integrate()
        mass_l = mass - data.error_bound
        mass_u = mass + data.error_bound  
        return mass_l,mass_u
    def _get_stdu_pts(self, n):
        random.seed(self.seed)
        if self.itype == 'SOBOL':
            return Sobol(self.d,seed=self.seed,graycode=True).gen_samples(n) # Sobol' samples from QMCPy. Could be replaced with IID samples
        elif self.itype == 'IID':
            return random.rand(n,self.d)
        else:
            raise Exception('init_type should be "Sobol" or "IID"')
    def _dlogpgt(self, x):
        ob_low = x < self.Lt
        ob_high = x > self.Ut
        ob = (ob_low+ob_high).max(1)
        ib = (~ob)[:,None]
        t = -(x@self.invSt)*ib - ob_high*self.beta/(x-self.Ut+1) + ob_low*self.beta/(self.Lt-x+1)
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
    def update(self, steps=100, epsilon=5e-3, eta=.5):
        """
        Update the samples. 
        If the dimensions are independent (covariance matrix is a diagnol matrix), 
        then there is no need to update samples, can just return samples directly

        Args:
            steps (int): number of iterations
            epsilon (float): step size
            eta (float): momentum hypterparameter for ADA gradient descent
        
        Return:
            ndarray: n x d array of samples mimicking the truncated distribuiton after taking another s steps
        """
        if self.independent:
            from warnings import warn
            msg = 'Dimensions are independent --> no need to update samples --> return exact samples'
            warn(msg)
            return self._scale_x(self.x)
        for b in range(self.blocks):
            i = b*self.n_block
            if b < (self.blocks-1):
                x = self.x[i:i+self.n_block]
                hgrad = self.hgrad[i:i+self.n_block]
                nb = self.n_block
            else: # last block, may have more than self.n_block samples
                x = self.x[i:]
                hgrad = self.hgrad[i:]
                nb = x.shape[0]
            for s in range(steps):
                ts = s + self.iter # total steps
                lnpgrad = self._dlogpgt(x)
                kxy,dxkxy = self._k_rbf(x)  
                grad = ((kxy@lnpgrad)+dxkxy)/nb  
                if ts==0:
                    hgrad += grad**2
                else:
                    hgrad = eta*hgrad+(1-eta)*(grad**2)
                adj_grad = grad/(self.fudge+sqrt(hgrad))
                x += epsilon*adj_grad 
                self.iter += 1
            self.x[i:i+nb] = x
            self.hgrad[i:i+nb] = hgrad
        self.iter += steps
        return self._scale_x(self.x)
    def reset(self):
        self.x = self.x_init.copy()
        self.iter = 0
        self.hgrad = zeros((self.n,self.d),dtype=float)
    def _scale_x(self, x):
        return x@diag(self.U_hat-self.L_hat) + self.mu
    def _get_cut_trunc(self, n_cut):
        x_stdu = self._get_stdu_pts(n_cut)
        evals,evecs = eigh(self.Sigma)
        order = argsort(-evals)
        A = dot(evecs[:,order],diag(sqrt(evals[order]))).T
        x_ut = norm.ppf(x_stdu)@A+self.mu
        x_cut = x_ut[(x_ut>self.L).all(1)&(x_ut<self.U).all(1)]
        return x_ut,x_cut
    def get_metrics(self, gn, gnt, verbose=True):
        x = self._scale_x(self.x)
        data = {
            'mu':{
                'TRUE': self.mu,
                'CUT': gnt.mean(0),
                'VITRUNC':x.mean(0)},
            'Sigma':{
                'TRUE': self.Sigma,
                'CUT': cov(gnt.T),
                'VITRUNC':cov(x.T)}}
        nOB = self.n-((x>self.L_hat).all(1)&(x<self.U_hat).all(1)).sum()
        g_l,g_u = self._approx_mass_qmc(self.L,self.U)
        mass_lost = 1-self.g_hat_l/g_l
        if verbose:
            set_printoptions(formatter={'float': lambda x: "{0:5.2f}".format(x)})
            for param,dd in data.items():
                print(param)
                for s,d in dd.items():
                    print('%15s: %s'%(s,str(d.flatten())))
            print('Points out of bounds:',nOB)
            print("mass lost: %.3f"%mass_lost)
        return data,nOB,mass_lost
    def plot(self, out=None, show=False):
        if self.d != 2: 
            msg = "`GTGS.plot` method only applicable when d=2"
            raise Exception(msg)
        # matplotlib metas
        from matplotlib import pyplot
        pyplot.rc('font', size=16)
        pyplot.rc('axes', titlesize=16, labelsize=16)
        pyplot.rc('xtick', labelsize=16)
        pyplot.rc('ytick', labelsize=16)
        pyplot.rc('legend', fontsize=16)
        pyplot.rc('figure', titlesize=16)
        # points
        gn,gnt = self._get_cut_trunc(self.n)
        x_init = self._scale_x(self.x_init)
        x = self._scale_x(self.x)
        # other params
        dpb0 = array([self.L_hat[0]-2,self.U_hat[0]+2])
        dpb1 = array([self.L_hat[1]-2,self.U_hat[1]+2])
        # plots
        fig,ax = pyplot.subplots(nrows=2,ncols=2,figsize=(15,15))
        self._plot_help(gn,ax[0,0],dpb0,dpb1,s=10,color='b',title="Points Before Truncation")
        self._plot_help(gnt,ax[0,1],dpb0,dpb1,s=10,color='b',title="Points with Cut Truncation",pltbds=True,lb=self.L,ub=self.U)
        self._plot_help(x_init,ax[1,0],dpb0,dpb1,s=10,color='b',title="Initial Points",pltbds=True,lb=self.L,ub=self.U)
        self._plot_help(x,ax[1,1],dpb0,dpb1,s=10,color='b',title="Final Points",pltbds=True,lb=self.L,ub=self.U)
        if out:
            pyplot.savefig(out,dpi=250)
        if show:
            pyplot.show()
        return fig,ax
    def _plot_help(self, x, ax, xlim, ylim, s, color, title, pltbds=False, lb=None, ub=None):
        ax.scatter(x[:,0],x[:,1],s=s,color=color)
        ax.set_xlim(xlim); ax.set_xticks(xlim); ax.set_xlabel('$x_{i,1}$')
        ax.set_ylim(ylim); ax.set_yticks(ylim); ax.set_ylabel('$x_{i,2}$')
        ax.set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0]))
        ax.set_title(title)
        if pltbds:
            ax.axhline(y=lb[1], color='k', linestyle='--')
            ax.axhline(y=ub[1], color='k', linestyle='--')
            ax.axvline(x=lb[0], color='k', linestyle='--')
            ax.axvline(x=ub[0], color='k', linestyle='--')

if __name__ == '__main__':
    gt = GT(
        n = 2**8, 
        d = 2,
        mu = [1,2], 
        Sigma = [[5,4],[4,9]], #[[5,0],[0,9]],
        L = [-4,-inf], 
        U = [inf,5], 
        init_type = 'Sobol',
        seed = None,
        n_block = None)
    gt.update(steps=1000, epsilon=5e-3, eta=.9)
    gt.plot(out='_ags.png', show=False)
    gn,gnt = gt._get_cut_trunc(2**20)
    gt.get_metrics(gn, gnt, verbose=True)
