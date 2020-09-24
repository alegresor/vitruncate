from vitruncate import GTGS,trunc_gen
from qmcpy import Sobol
from numpy import *
from numpy.linalg import inv
from scipy import stats
from matplotlib import pyplot

pyplot.rc('font', size=16)
pyplot.rc('axes', titlesize=16, labelsize=16)
pyplot.rc('xtick', labelsize=16)
pyplot.rc('ytick', labelsize=16)
pyplot.rc('legend', fontsize=16)
pyplot.rc('figure', titlesize=16)

def scat_plot(x,ax,xlim,ylim,s,color,title,pltbds=False,lb=None,ub=None):
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

def plt_sobol():
    n,d = 2**8,2
    s = Sobol(d,seed=7)
    u = s.gen_samples(n)
    fig,ax = pyplot.subplots(nrows=1,ncols=1,figsize=(5,5))
    scat_plot(u,ax,[0,1],[0,1],s=10,color='r',title="Sobol' Points")

def plt_std_gt():
    n,d = 2**8,2
    s = Sobol(d,seed=7)
    u = s.gen_samples(n)
    # parameters
    trunc_lb = array([-1,-2])
    trunc_ub = array([1,2])
    # points
    g = trunc_gen(u,tile(-inf,d),tile(inf,d),stats.norm)
    gt = trunc_gen(u,trunc_lb,trunc_ub,stats.norm)
    t = trunc_gen(u,tile(-inf,d),tile(inf,d),stats.t,df=tile(5,d))
    tt = trunc_gen(u,trunc_lb,trunc_ub,stats.t,df=tile(5,d))
    # plots
    fig,ax = pyplot.subplots(nrows=1,ncols=4,figsize=(25,5))
    scat_plot(g,ax[0],[-3,3],[-3,3],s=10,color='b',title="Non-Truncated Gaussian Points")
    scat_plot(gt,ax[1],[-3,3],[-3,3],s=10,color='b',title="Truncated Gaussian Points",pltbds=True,lb=trunc_lb,ub=trunc_ub)
    scat_plot(t,ax[2],[-3,3],[-3,3],s=10,color='g',title="Non-Truncated Student Points")
    scat_plot(tt,ax[3],[-3,3],[-3,3],s=10,color='g',title="Truncated Student Points",pltbds=True,lb=trunc_lb,ub=trunc_ub)

def plt_non_std_gt():
    n,d = 2**8,2
    s = Sobol(d,seed=7)
    u = s.gen_samples(n)
    # parameters
    mu = array([1,2])
    Sigma = array([1,4])
    trunc_lb = [-1,-2]
    trunc_ub = [3,6]
    # points
    g = trunc_gen(u,tile(-inf,d),tile(inf,d),stats.norm,loc=mu,scale=sqrt(Sigma))
    gt = trunc_gen(u,trunc_lb,trunc_ub,stats.norm,loc=mu,scale=sqrt(Sigma))
    t = trunc_gen(u,tile(-inf,d),tile(inf,d),stats.t,df=tile(5,d),loc=mu,scale=sqrt(Sigma))
    tt = trunc_gen(u,trunc_lb,trunc_ub,stats.t,df=tile(5,d),loc=mu,scale=sqrt(Sigma))
    # plots
    fig,ax = pyplot.subplots(nrows=1,ncols=4,figsize=(25,5))
    scat_plot(g,ax[0],[-3,5],[-4,8],s=10,color='b',title="Non-Truncated Gaussian Points")
    scat_plot(gt,ax[1],[-3,5],[-4,8],s=10,color='b',title="Truncated Gaussian Points",pltbds=True,lb=trunc_lb,ub=trunc_ub)
    scat_plot(t,ax[2],[-3,5],[-4,8],s=10,color='g',title="Non-Truncated Student Points")
    scat_plot(tt,ax[3],[-3,5],[-4,8],s=10,color='g',title="Truncated Student Points",pltbds=True,lb=trunc_lb,ub=trunc_ub)

def plt_gtgs():
    # parameters
    mu = array([1,2],dtype=float)
    Sigma = array([[9,4],[4,5]],dtype=float)
    #L = array([-4,-3],dtype=float)
    #U = array([6,6],dtype=float)
    L = array([-inf,-inf],dtype=float)
    U = array([inf,inf],dtype=float)
    epsilon = 1e-2
    walk1 = 500
    walk2 = 500
    # points
    #    1
    evals,evecs = linalg.eigh(Sigma)
    order = argsort(-evals)
    A = dot(evecs[:,order],diag(sqrt(evals[order]))).T
    s = Sobol(2,seed=7)
    u = s.gen_samples(2**8)
    x = stats.norm.ppf(u)@A+mu
    #    2
    gtgs = GTGS(2**8,2,mu,Sigma,L,U,epsilon)
    x_init = gtgs.get_val()
    #    3
    xw1 = gtgs.walk(walk1)
    print('Walk 1 mean:',xw1.mean(0))
    #    4
    xw2 = gtgs.walk(walk2)
    print('Walk 2 mean:',xw2.mean(0))
    # plots
    fig,ax = pyplot.subplots(nrows=1,ncols=4,figsize=(25,5))
    scat_plot(x,ax[0],[-7,9],[-5,9],s=10,color='b',title="Sample of Points Before Truncation")
    scat_plot(x_init,ax[1],[-7,9],[-5,9],s=10,color='b',title="Initial Points for Stein Method",pltbds=True,lb=L,ub=U)
    scat_plot(xw1,ax[2],[-7,9],[-5,9],s=10,color='b',title="Points After %d Iterations"%walk2,pltbds=True,lb=L,ub=U)
    scat_plot(xw2,ax[3],[-7,9],[-5,9],s=10,color='b',title="Points After Another %d Iterations"%walk2,pltbds=True,lb=L,ub=U)

if __name__ == '__main__':
    #plt_sobol()
    #plt_std_gt()
    #plt_non_std_gt()
    plt_gtgs()

    pyplot.savefig('_ags.png')