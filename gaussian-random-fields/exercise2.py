import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample
import pandas as pd
import random
from scipy.spatial import distance
from scipy import linalg

def create_samples(m,sigma,eta,tau,alpha):
    x = np.zeros((m , 2))
    x[:,0] = np.random.uniform(0,1,m)
    x[:,1] = np.random.uniform(0,1,m)
    S = pd.DataFrame({'Easting' : x[:,0], 'Northing': x[:,1]} , columns=['Easting' , 'Northing'])
    spatial_cov = np.zeros((m , m))
    for i in range(m):
        for j in range(m):
            tij = abs(distance.euclidean(x[i,:] , x[j,:]))
            spatial_cov[i , j] =  (sigma**2) * (1 + eta * tij) * np.exp(-(eta * tij))
    L = linalg.cholesky(spatial_cov, lower=True) 
    U = np.random.normal(0, 1, len(L))
    tmp = L @ U
    h = (x[:,0] + x[:,1] - 1)
    tmp += alpha*h
    y = tmp + np.random.normal(0,tau)
    S['Y'] = y
    return(x,y,h,S)


# calculate the partial derivatives of covariance matrix
def partial_derivative(theta, x): # theta = [sigma^2, eta, tau^2],
    tau_squared_partial = np.identity(m)
    sigma_squared_partial = np.identity(m)
    eta_partial = np.zeros((m, m))

    for i in range(1, m):
        for j in range(i+1, m):
            t_ij = abs(distance.euclidean(x[i,1:],x[j,1:]))
            sigma_squared_partial[i, j] = (1+theta[1]* t_ij)*np.exp(-theta[1]*t_ij)
            sigma_squared_partial[j, i] = sigma_squared_partial[i, j]

            eta_partial[i, j] = -theta[0]*theta[1]*(t_ij**2)*np.exp(-theta[1]*t_ij)
            eta_partial[j, i] = eta_partial[i, j]

    return np.concatenate((sigma_squared_partial.reshape(m, m, 1),
                eta_partial.reshape(m, m, 1), tau_squared_partial.reshape(m, m, 1)), axis = 2)


# log-likelihood calculation
def calc_likelihood(C, Q, h, y, beta):
    return(-(1/2)*np.log(np.linalg.det(C))-(1/2)*np.transpose(y-h*beta)@Q@(y-h*beta))  

# calculate covariance matrix
def calc_C(theta,x):
    C = np.identity(m)*(theta[0]+theta[2])
    for i in range(1, m):
        for j in range(i+1, m):
            t_ij = abs(distance.euclidean(x[i,1:], x[j,1:]))
            C[i, j] = theta[0]*(1+theta[1]*t_ij)*np.exp(-theta[1]*t_ij)
            C[j, i] = C[i, j]
    Q = np.linalg.inv(C)
    return(C,Q)

# Fisher Scoring for optimization of log-likelihood
def make_score(Q, h, x, y, theta, beta):
    part_vec = partial_derivative(theta, x)
    likelihood_derivative = np.zeros(3)
    score_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(i,3):
            score_matrix[i, j] = -(1/2)*np.trace((Q@part_vec[:,:,i]@Q)@part_vec[:,:,j])
            score_matrix[j ,i] = score_matrix[i, j]
        likelihood_derivative[i] = -(1/2)*np.trace(Q@part_vec[:,:,i])+(1/2)*np.transpose(y-beta*h)@(Q@part_vec[:,:,i]@Q)@(y-beta*h)
    
    return(np.linalg.inv(score_matrix)@likelihood_derivative)



def newton_raphson(x,h,y):
    theta = np.array([2, 5, 2])
    beta = 2
    tol = 0.001
    rho = 1000
    prev_likelihood = 1000
    likelihood = 100
    while rho > tol:
        C, Q = calc_C(theta,x)
        beta = 1/(np.transpose(h)@Q@h)*np.transpose(h)@Q@y
        theta = theta-make_score(Q, h, x, y, theta, beta)
        prev_likelihood = likelihood
        likelihood = calc_likelihood(C, Q, h, y, beta)
        rho = abs(likelihood - prev_likelihood)
    return(beta, theta)
    
if __name__ == "__main__":
  m = 200 
  sigma = 1 
  eta = 10 
  tau = 0.05 
  alpha = 1
  x,y,h,S = create_samples(m,sigma,eta,tau,alpha)
  beta, theta = newton_raphson(x,h,y)
