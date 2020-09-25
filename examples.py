from vitruncate import GTGS
from numpy import *

if __name__ == '__main__':
    gt = GTGS(
        n = 2**8, 
        d = 2,
        mu = [1,2], 
        Sigma = [[5,4],[4,9]], 
        L = [-4,-3], 
        U = [6,6], 
        init_type = 'Sobol',
        seed = None)
    x = gt.update(
        steps = 100,
        epsilon =5e-3,
        alpha = .5)
    gt.plot(
        verbose = True,
        out = '_ags.png',
        show = False)
    