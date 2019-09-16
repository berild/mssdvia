
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

N = 100
t = 0.001

A = np.zeros((N, N))
x = np.linspace(0, 0.99, 100)
for i in range(N):
    for j in range(N):
        d = min(np.abs(x[i]-x[j]), 1-np.abs(x[i]-x[j]))
        A[i,j] = (0.01)/np.sqrt(4*np.pi*t)*np.exp(-d**2/(4*t))


#print(A)
y = pd.read_csv("OppgA.txt",header=None)
y = np.array(y)

x_hat = np.linalg.solve(A,y)
#print(np.linalg.solve(A,y))
plt.plot(x_hat)
plt.savefig('x_hat.pdf')
plt.show()
#print(np.linalg.eig(A))
plt.plot(np.linalg.eig(A)[0])
plt.show()

# Add plot of eigenvalues


# b)

# Computing the posterior expectation and variance
#print(len(np.identity(N)))
A_T = np.matrix.transpose(A)
Sigma_e = 0.25*np.identity(100)
Sigma_x = np.identity(100)
mu_x = np.zeros(100)
y_T = np.matrix.transpose(y)
Sigma_e_inv = np.linalg.inv(Sigma_e)
Sigma_x_inv = np.linalg.inv(Sigma_x)
mu_x_T = np.matrix.transpose(mu_x)
eta = np.matmul(np.matmul(y_T, Sigma_e_inv),A) + np.matmul(mu_x_T,Sigma_x_inv)

Sigma_inv = np.matmul(np.matmul(A_T,Sigma_e_inv),A) + Sigma_x_inv

Sigma = np.linalg.inv(Sigma_inv)

mu_posterior_b = np.matmul(Sigma,np.matrix.transpose(eta))

std_dev = np.sqrt(np.diagonal(Sigma))

lower_bound = mu_posterior_b - 1.96*std_dev
upper_bound = mu_posterior_b + 1.96*std_dev

plt.figure()
plt.plot(lower_bound)
plt.plot(upper_bound)
plt.plot(mu_posterior_b)
plt.plot(y)
plt.show()


print(mu_posterior_b)
# c)


Sigma_x = np.zeros((N, N))


for i in range(N):
    for j in range(N):
        d = min(abs(x[i]-x[j]),1-abs(x[i]-x[j]))
        Sigma_x[i,j] = np.exp(-d/0.1)
        
# Computing the posterior expectation and variance
#print(len(np.identity(N)))
A_T = np.matrix.transpose(A)
Sigma_e = 0.25*np.identity(100)
#Sigma_x = np.identity(100)
mu_x = np.zeros(100)
y_T = np.matrix.transpose(y)
Sigma_e_inv = np.linalg.inv(Sigma_e)
Sigma_x_inv = np.linalg.inv(Sigma_x)
mu_x_T = np.matrix.transpose(mu_x)
eta = np.matmul(np.matmul(y_T, Sigma_e_inv),A) + np.matmul(mu_x_T,Sigma_x_inv)

Sigma_inv = np.matmul(np.matmul(A_T,Sigma_e_inv),A) + Sigma_x_inv

Sigma = np.linalg.inv(Sigma_inv)

mu_posterior_b = np.matmul(Sigma,np.matrix.transpose(eta))

std_dev = np.sqrt(np.diagonal(Sigma))

lower_bound = mu_posterior_b - 1.96*std_dev
upper_bound = mu_posterior_b + 1.96*std_dev

plt.figure()
plt.plot(lower_bound)
plt.plot(upper_bound)
plt.plot(mu_posterior_b)
plt.plot(y)
plt.show()



