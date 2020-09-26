from numpy import *
from numpy.linalg import *
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
        
class GT(object):
    """
    Gaussian Truncated Distribution Generator by Stein Method.
    Code adapted from: https://github.com/DartML/Stein-Variational-Gradient-Descent/blob/master/python/svgd.py
    """
    def __init__(self, n, d, mu, Sigma, L, U, init_type='IID', seed=None):
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
        """
        self.n = n
        self.d = d
        self.mu = array(mu,dtype=float)[:,None]
        self.Sigma = array(Sigma,dtype=float)
        self.independent = (self.Sigma==(self.Sigma*eye(self.d))).all()
        self.sn = det(self.Sigma) # normalization factor
        self.Sigma_s = Sigma/self.sn
        self.invSigma = inv(self.Sigma_s)
        self.L = array(L).flatten()
        self.U = array(U).flatten()
        self.B = (vstack((self.L,self.U))-self.mu.T)/sqrt(self.sn) # 2 x d array of bounds
        self.seed = seed
        self.itype = init_type.upper()
        self.x_stdu = self._get_stdu_pts(self.n)  
        self.x_init = zeros((self.n,self.d),dtype=float)
        for j in range(self.d):
            std = sqrt(self.Sigma_s[j,j])
            cdflb = norm.cdf(self.B[0,j],loc=0,scale=std)
            cdfub = norm.cdf(self.B[1,j],loc=0,scale=std)
            self.x_init[:,j] = norm.ppf((cdfub-cdflb)*self.x_stdu[:,j]+cdflb,loc=0,scale=std)
        self.x = self.x_init.copy()
        self.fudge = 1e-6
        self.iter = 0
        self.hgrad = zeros((self.n,self.d),dtype=float)
    def _get_stdu_pts(self, n):
        random.seed(self.seed)
        if self.itype == 'SOBOL':
            from qmcpy import Sobol
            return Sobol(self.d,seed=self.seed,graycode=True).gen_samples(n) # Sobol' samples from QMCPy. Could be replaced with IID samples
        elif self.itype == 'IID':
            return random.rand(n,self.d)
        else:
            raise Exception('init_type should be "Sobol" or "IID"')
    def _dlogpgt(self, x):
        maxd = 1e15
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
    def update(self, steps=100, epsilon=5e-3, alpha=.5):
        """
        Update the samples. 
        If the dimensions are independent (covariance matrix is a diagnol matrix), 
        then there is no need to update samples, can just return samples directly

        Args:
            steps (int): number of iterations
            epsilon (float): step size
            alpha (float): momentum hypterparameter for ADA gradient descent
        
        Return:
            ndarray: n x d array of samples mimicking the truncated distribuiton after taking another s steps
        """
        if self.independent:
            from warnings import warn
            msg = 'Dimensions are independent --> no need to update samples --> return exact samples'
            warn(msg)
            return self._scale_x(self.x)
        for i in range(steps):
            lnpgrad = self._dlogpgt(self.x)
            kxy,dxkxy = self._k_rbf(self.x)  
            grad = ((kxy@lnpgrad)+dxkxy)/self.n  
            if self.iter==0:
                self.hgrad += grad**2
            else:
                self.hgrad = alpha*self.hgrad+(1-alpha)*(grad**2)
            adj_grad = grad/(self.fudge+sqrt(self.hgrad))
            self.x += epsilon*adj_grad 
            self.iter += 1
        return self._scale_x(self.x)
    def reset(self):
        self.iter = 0
        self.x = self.x_init
        self.hgrad = zeros((self.n,self.d),dtype=float)
    def _scale_x(self, x):
        return x*sqrt(self.sn)+self.mu.T
    def _get_cut_trunc(self, n_cut):
        x_stdu = self._get_stdu_pts(n_cut)
        evals,evecs = eigh(self.Sigma)
        order = argsort(-evals)
        A = dot(evecs[:,order],diag(sqrt(evals[order]))).T
        x_ut = norm.ppf(x_stdu)@A+self.mu.T
        x_cut = x_ut[(x_ut>self.L).all(1)&(x_ut<self.U).all(1)]
        return x_ut,x_cut
    def get_metrics(self, gn, gnt, verbose=True):
        x = self._scale_x(self.x)
        data = {
            'mu':{
                'TRUE': self.mu.flatten(),
                'CUT': gnt.mean(0),
                'VITRUNC':x.mean(0)},
            'Sigma':{
                'TRUE': self.Sigma,
                'CUT': cov(gnt.T),
                'VITRUNC':cov(x.T)}}
        nOB = self.n-((x>self.L).all(1)&(x<self.U).all(1)).sum()
        if verbose:
            set_printoptions(formatter={'float': lambda x: "{0:5.2f}".format(x)})
            for param,dd in data.items():
                print(param)
                for s,d in dd.items():
                    print('%15s: %s'%(s,str(d.flatten())))
            print('Points out of bounds:',nOB)
        return data,nOB
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
        dpb0 = array([self.L[0]-2,self.U[0]+2])
        dpb1 = array([self.L[1]-2,self.U[1]+2])
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
        ax.set_xlim(xlim);ax.set_xticks(xlim);ax.set_xlabel('$x_{i,1}$')
        ax.set_ylim(ylim);ax.set_yticks(ylim);ax.set_ylabel('$x_{i,2}$')
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
        Sigma = [[5,4],[4,9]], # [[5,0],[0,9]]
        L = [-4,-3], 
        U = [6,6], 
        init_type = 'IID',
        seed = None)
    gt.update(steps=5000, epsilon=5e-3, alpha=.5)
    gt.plot(out='_ags.png', show=False)
    gn,gnt = gt._get_cut_trunc(2**20)
    gt.get_metrics(gn, gnt, verbose=True)