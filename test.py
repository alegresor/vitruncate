from vitruncate import GT
from numpy import * 

def test(gt, strides, trials, n_cut, epsilon, alpha):
    error = lambda p,p_hat: abs(p.flatten()-p_hat.flatten()).sum() # abs error
    gn,gnt = gt._get_cut_trunc(n_cut)
    mu_errors = zeros((trials,len(strides)),dtype=float)
    Sigma_errors = zeros((trials,len(strides)),dtype=float)
    nOBs = zeros((trials,len(strides)),dtype=float)
    for t in range(trials):
        print('Trial #',t+1)
        gt.reset()
        for s in range(len(strides)):
            if (s+1)%10==0: print('\tStride #',s+1)
            gt.update(strides[s],epsilon,alpha)
            data,nOB = gt.get_metrics(gn,gnt,verbose=False)
            mu_errors[t,s] = error(data['mu']['CUT'],data['mu']['VITRUNC'])
            Sigma_errors[t,s] = error(data['Sigma']['CUT'],data['Sigma']['VITRUNC'])
            nOBs[t,s] = nOB
    from pandas import DataFrame
    df = DataFrame({
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
        L = [-4,-3], 
        U = [6,6], 
        init_type = 'Sobol',
        seed = None)
    df = test(gt, strides=tile(25,40), trials=10, n_cut=2**22, epsilon=5e-3, alpha=.5)
    print(df)
    df.to_csv('out/test.csv')
