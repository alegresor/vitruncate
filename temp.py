from scipy.stats import multivariate_normal as mvn 
from scipy.linalg import *
from numpy import *
from math import comb

# constants
d = 3
mu = array([[1],[2],[3]])
Sigma = array([[5,4,3],[4,9,2],[3,2,8]])
L = array([[-4],[-3],[-1]])
U = array([[6],[6],[6]])

# converstion matricies
W_inv = diag(1/(U-L).flatten())
L_new = W_inv@(L-mu)
U_new = W_inv@(U-mu)
mu_new = mu-mu
Sigma_new = W_inv@Sigma@W_inv.T

# numerical tests

evals,evecs = eigh(Sigma_new)
order = argsort(-evals)
A = dot(evecs[:,order],diag(sqrt(evals[order]))).T
n = 10000
x_01box = random.randn(n,d)@A
x_ogbox = x_01box@diag((U-L).flatten())+mu.flatten()

# new pdf
alpha = .01 # 99% of mass

G = mvn.cdf(U.flatten(),mu.flatten(),Sigma) - mvn.cdf(L.flatten(),mu.flatten(),Sigma)
coefs = zeros(d+1,dtype=double)
coefs[0] = alpha*G/(1-alpha)
for j in range(1,d+1):
    coefs[j] = -comb(d,j)*2**j

r = roots(coefs)
beta = real(r.max())+1

print(coefs)
print(r)
print(beta)