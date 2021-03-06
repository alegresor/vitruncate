from vitruncate import GT
from numpy import *
import pandas as pd

def test(gt, strides, trials, n_cut, epsilon, eta):
    error = lambda p,p_hat: ((p.flatten()-p_hat.flatten())**2).mean() # abs error
    gn,gnt = gt._get_cut_trunc(n_cut)
    mu_errors = zeros((trials,len(strides)),dtype=float)
    Sigma_errors = zeros((trials,len(strides)),dtype=float)
    nOBs = zeros((trials,len(strides)),dtype=float)
    for t in range(trials):
        print('Trial #',t+1)
        gt.reset()
        for s in range(len(strides)):
            if (s+1)%10==0: print('\tStride #',s+1)
            gt.update(strides[s],epsilon,eta)
            data,nOB,mass_lost = gt.get_metrics(gn,gnt,verbose=False)
            mu_errors[t,s] = error(data['mu']['CUT'],data['mu']['VITRUNC'])
            Sigma_errors[t,s] = error(data['Sigma']['CUT'],data['Sigma']['VITRUNC'])
            nOBs[t,s] = nOB
    df = pd.DataFrame({
        'steps': cumsum(strides),
        'mu_error': mu_errors.mean(0),
        'Sigma_error': Sigma_errors.mean(0),
        'OutOfBounds': nOBs.mean(0)})
    return df

if __name__ == '__main__':
    gt = GT(
        n = 2**8, 
        d = 2,
        mu = [1,2], 
        Sigma = [[5,4],[4,9]], 
        L = [-5,-3], 
        U = [6,4], 
        init_type = 'Sobol',
        seed = None)
    df = test(gt, strides=tile(1,50), trials=5, n_cut=2**22, epsilon=5e-3, eta=.5)
    df.to_csv('converge/converge.csv')
    df = pd.read_csv('converge/converge.csv')
    print(df)
    from matplotlib import pyplot
    fig,ax = pyplot.subplots()
    ax.plot(df['steps'],df['mu_error'],color='b',label='$\mu$')
    ax.plot(df['steps'],df['Sigma_error'],color='g',label='$\Sigma$ ')
    ax.set_xlim([df['steps'].min(),df['steps'].max()])
    ax.set_ylim([0,max(df['mu_error'].max(),df['Sigma_error'].max())])
    ax.legend()
    ax.set_xlabel('steps')
    ax.set_ylabel('RMSE')
    pyplot.savefig('converge/converge.png',dpi=250)
    #pyplot.show()
