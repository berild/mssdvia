library(ggplot2)
library(fields)

# a)

n = 50 # number of railroad sections
p_0 = 0.99 # p = (x_1 = 0)
r = 0.95 # k = p(x_i+1 = 0 | x_i = 0)
tau = 0.3 # standard deviation in likelihood model
prior_prob = rep(0,n)
for (i in 1:n){
  prior_prob[i] = 1-p_0*r**(i-1)
}
prior_x = rep(0,n)
for (i in 1:n){
  if (prior_prob[i] < 0.5){
    prior_x[i] = 0
  }
  else{
    prior_x[i] = 1
  }
}

plot(c(1:n),prior_prob,ylab="prob",xlab="regions",ylim=c(0,1))


# b)


PV = max(-100000, -5000*(do.call(sum,list(prior_prob))))

# Optimal decision: clean ahead

# c)

set.seed(2019)

forward_backward <- function(y, r, tau_vec){
  alpha = rep(0,n) # p(x_i = 0 | y_1, ... , y_i)
  alpha[1] = dnorm(y[1],0,tau_vec[1])*p_0/(dnorm(y[1],0,tau_vec[1])*p_0+dnorm(y[1],1,tau_vec[1])*(1-p_0))
  temp_0 = rep(0,n) # p(x_i = 0| y_1, ... y_i-1)
  temp_2 = rep(0,n) # p(y_i | y_1, ... y_i-1)

  for (i in 2:n){
    temp_0[i] = r * alpha[i - 1]
    temp_2[i] = dnorm(y[i],0,tau_vec[i])*temp_0[i]+dnorm(y[i],1,tau_vec[i])*(1-temp_0[i])
    alpha[i] = dnorm(y[i], 0, tau_vec[i]) * temp_0[i] / temp_2[i]
  }
  
  cond_prob = rep(0,n) # p(x_i = 1 | y_1, ... y_n)
  
  #temp_4 = np.zeros(n) # p(x_i = 1 | y_1, ... , y_i+1, x_i+1 = 0)
  temp_5 = rep(0,n) # p(x_i = 1 | y_1, ... , y_i+1, x_i+1 = 1)
  cond_prob[n] = 1-alpha[n]
  for (i in (n-1):1){
    #temp_4[i] = P[1][0]*(1-alpha[i])/temp_0[i+1]
    temp_5[i] = (1-alpha[i])/(1-temp_0[i+1])

  cond_prob[i] = temp_5[i]*cond_prob[i+1] #temp_4[i]*(1-cond_prob[i+1])+temp_5[i]*cond_prob[i+1]
  }  
  return (cond_prob)
}


task_c <- function(){
  prior_x = rep(0.5, n)
  prior_x[1] = rbinom(1, 1, marg_prob[1])
  
  for (i in 2:n){
    if (prior_x[i-1] == 1){prior_x[i] = 1}
    else{
      if (runif(1)<r){prior_x[i] = 0}
      else {prior_x[i] = 1}
    }
  }
  y_vec = rep(0.5,n)
  y_vec[20] = 0.2
  y_vec[30] = 0.7
  tau_vec = 100 * tau * rep(1,n)
  tau_vec[20] = tau
  tau_vec[30] = tau
  state_prob = forward_backward(y_vec, r, tau_vec)
  #par(new=TRUE)
  plot(1:n, state_prob,ylab="prob",xlab="regions",ylim=c(0,1))
  
  return (state_prob)
}

state = task_c()


# d)

PoV_k <- function(k, B, tau){
  PoV = 0
  tau_vec = 100*tau*rep(1,n)
  tau_vec[k] = tau
  prior_x = 0.5 * rep(1,n)
  for (i in 1:B){
    prior_x[k] = rbinom(n=1, size=1, prob = prior_prob[k])
    y_vec = rnorm(n = n, prior_x, tau_vec)
    marg_prob = forward_backward(y_vec, r, tau_vec)
    
    PoV = PoV + max(-100000, -5000*(do.call(sum, list(marg_prob))))
  }
  print(k)
  return (PoV/B)
}

task_d <- function(B){
  PoV_values_k = rep(0,n)
  for (k in 1:n){
    PoV_values_k[k] = PoV_k(k, B, tau)
  }

  VOI_1 = PoV_values_k - PV
  plot(1:n, VOI_1)
  print(VOI_1)
  print(which.max(VOI_1))
  print(max(VOI_1))
}


B = 10000
task_d(B)


# B = 10000, VOI = 11278.02, x = 31



# e)


PoV_D <- function(k_1, k_2, B, tau, prior_prob, r){
  PoV = 0
  tau_vec = 100*tau*rep(1,n)
  prior_x = rep(0.5, n)
  
  if (k_1 == k_2){
    for (i in 1:B){
      y_vec = rnorm(n, prior_x, tau_vec)
      prior_x[k_1] = rbinom(1, 1, prior_prob[k_1])
      y_vec[k_1] = rnorm(n=1,prior_x[k_1], tau/sqrt(2))
      tau_vec[k_1] = tau
      
      marg_prob = forward_backward(y_vec, r, tau_vec)
      PoV = PoV + max(-100000, -5000*(do.call(sum, list(marg_prob))))   
    }
  }
  else{
    for (i in 1:B){
      y_vec = rnorm(n, prior_x, tau_vec)
      prior_x[k_1] = rbinom(1, 1, prior_prob[k_1])
      for (i in (k_1+1):k_2)
        if (prior_x[i-1] == 1){prior_x[i] = 1}
        else{
          if (runif(1)<r){prior_x[i] = 0}
          else {prior_x[i] = 1}
        }
      y_vec[k_1] = rnorm(1, prior_x[k_1],tau) 
      y_vec[k_2] = rnorm(1, prior_x[k_2],tau)
      tau_vec[k_1] = tau
      tau_vec[k_2] = tau
      marg_prob = forward_backward(y_vec, r, tau_vec)
      PoV = PoV + max(-100000, -5000*(do.call(sum, list(marg_prob))))   
    }
  }  
    
  return (PoV/B)
}
  


task_e <- function(B, tau, prior_prob, r){
  regions_PoV = matrix(0, nrow = n, ncol = n)

  for (i in 1:n){
    for (j in i:n){
      cat(i, " ",j, "\n")
      if (i == j){regions_PoV[i,i] = PoV_D(i, i, 10*B, tau, prior_prob, r)} # Diagonal
      else{ # Off-diagonal 
        regions_PoV[i,j] = PoV_D(i, j, B, tau, prior_prob, r) 
        regions_PoV[j,i] = regions_PoV[i,j]
      }
    }
  }
  return (regions_PoV)
}
B = 1000
PoV_e = task_e(B, tau, prior_prob, r)
VoI_e = PoV_e - PV
image.plot(x = 1:50, y = 1:50, z = VoI_e, xlab = "i", ylab = "j")

which(VoI_e == max(VoI_e), arr.ind = TRUE)
max(VoI_e)

# B = 1000, best points: (21, 33), (33, 21), VOI = 14810.96

# f)


PoV_all <- function(B, tau, prior_prob,r){
  PoV = 0
  tau_vec = tau*rep(1,n)
  prior_x = rep(0.5,n)
  for (i in 1:B){
    prior_x[1] = rbinom(n=1, size=1, p=prior_prob[1])
    for (i in 2:n){
      if (prior_x[i-1] == 1){prior_x[i] = 1}
      else{
        if (runif(1)<r){prior_x[i] = 0}
        else {prior_x[i] = 1}
      }
    }
    y_vec = rnorm(n=n,prior_x, tau_vec)
    marg_prob = forward_backward(y_vec, r, tau_vec)
    PoV = PoV + max(-100000, -5000*(do.call(sum, list(marg_prob))))
  }
  return (PoV/B)
}


task_f <- function(B){

  tau = 1
  PoV_value_3 = PoV_all(B, tau, prior_prob,r)
  
  
  VOI_3 = PoV_value_3 - PV
  print(VOI_3)
}
B = 50000
task_f(B)

# VOI_3 =  12830.01



