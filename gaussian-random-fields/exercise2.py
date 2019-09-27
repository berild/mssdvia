import numpy as np
from scipy.spatial import distance
from scipy import linalg
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample

if __name__ == "__main__":
  m = 200 
  sigma = 1 
  eta = 10 
  tau = 0.05 
  alpha = 1
  y, x = create_samples(m,sigma,eta,tau,alpha)


def create_samples(m,sigma,eta,tau,alpha):
  x = np.ones((m,3))
  x[:,1] = np.random.uniform(0,1,m)
  x[:,2] = np.random.uniform(0,1,m)
  spatial_cov = np.zeros((m , m))
  for i in range(m):
    for j in range(m):
        tij = abs(distance.euclidean(x[i,1:] , x[j,1:]))
        spatial_cov[i , j] =  (sigma**2) * (1 + eta * tij) * np.exp(-(eta * tij))
  L = linalg.cholesky(spatial_cov, lower=True) 
  U = np.random.normal(0, 1, len(L))
  tmp = L @ U
  tmp += alpha*(x[:,1] + x[:,2] - 1)
  y = tmp + np.random.normal(0,tau)
  return(y,x)
