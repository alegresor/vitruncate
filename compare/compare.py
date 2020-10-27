from vitruncate import GT
from numpy import *
from time import time

for n in [100]:#[10000,100000]:
    t0 = time()
    gt = GT(
        n = n, 
        d = 2,
        mu = [0,0], 
        Sigma = [[5,4],[4,9]],
        L = [-1,-1], 
        U = [1,1], 
        init_type = 'Sobol',
        seed = None)
    gt.update(steps=50, epsilon=5e-3, eta=.9)
    print('%10d samples time: %.2f'%(n,time()-t0))