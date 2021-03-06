from vitruncate import GT
from numpy import *
from time import time

for n in [10000,100000]:
    t0 = time()
    gt = GT(
        n = n, 
        d = 2,
        mu = [0,0], 
        Sigma = [[5,4],[4,9]],
        L = [-1,-1], 
        U = [1,1], 
        init_type = 'IID',
        seed = None,
        n_block = 200)
    gt.update(steps=50, epsilon=5e-3, eta=.9)
    print('%10d samples time: %.2f'%(n,time()-t0))

# current best:
"""
 10000 samples time: 2.30
100000 samples time: 24.93
"""