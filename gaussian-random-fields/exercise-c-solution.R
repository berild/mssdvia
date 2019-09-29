library(tidyverse)

create_samples<-function(m,sigma,eta,tau,alpha){
  x = matrix(0, nrow = m,ncol = 2)
  x[,1] = rnorm(m)
  x[,2] = rnorm(m)
  S = data.frame(easting = x[,1], northing = x[,2])
  spatial_cov = diag(m)*sigma
  for (i in seq(m-1)){
    for (j in seq(i+1,m)){
      tij = distance(x[i,],x[j,])
      spatial_cov[i,j] = spatial_cov[j,i] = sigma*(1 + eta*tij)*exp(-eta*tij)
    }
  }
  L = t(chol(spatial_cov))
  U = rnorm(m)
  h = (x[,1] + x[,2] - 1)
  tmp = L%*%U
  y = tmp + alpha*h + rnorm(m, mean = 0, sd = sqrt(tau))
  return(list(
    x = x,
    y = y[,1],
    h = h,
    S = S,
    spatial_cov = spatial_cov
  ))
}

distance <- function(x,y){
  sqrt((y[1]-x[1])^2 + (y[2]-x[2])^2)
}

part.deriv <- function(theta,x,m){
  tau_part = diag(m)
  sigma_part = diag(m)
  eta_part = matrix(0, ncol = m, nrow = m)
  for (i in seq(m-1)){
    for (j in seq(i+1,m)){
      tij = distance(x[i,],x[j,])
      sigma_part[i,j] = sigma_part[j,i] = (1 + theta[2]*tij)*exp(-theta[2]*tij)
      eta_part[i,j] = eta_part[j,i] = -theta[1]*theta[2]*tij^2*exp(-theta[2]*tij)
    }
  }
  return(list(
    sigma_part = sigma_part,
    eta_part = eta_part,
    tau_part = tau_part
  ))
}

log.like <- function(c_mat,Q,h,y,beta,m){
  m/2*log(2*pi) -1/2*log(det(c_mat))-(1/2)*t(y-beta*h)%*%Q%*%(y-beta*h)
}

calc.c <- function(theta,x,m){
  c_mat = diag(m)*(theta[1] + theta[3])
  for (i in seq(m-1)){
    for (j in seq(i+1,m)){
      tij = distance(x[i,],x[j,])
      c_mat[i,j] = c_mat[j,i] = theta[1]*(1 + theta[2]*tij)*exp(-theta[2]*tij)
    }
  }
  Q = solve(c_mat)
  return(list(
    c_mat = c_mat,
    Q = Q
  ))
}

fish.score <- function(Q,h,x,y,theta,beta,m){
  part_list = part.deriv(theta,x,m)
  like_deriv = numeric(3)
  score_mat = matrix(0,nrow = 3,ncol  = 3)
  for (i in  seq(3)){
    for (j in seq(3)){
      score_mat[i,j] = -1/2*sum(diag(Q%*%part_list[[j]]%*%Q%*%part_list[[i]]))
    }
    like_deriv[i] = -1/2*sum(diag(Q%*%part_list[[i]])) + 
      1/2*t(y-beta*h)%*%(Q%*%part_list[[i]]%*%Q)%*%(y-beta*h)
  }
  return(solve(score_mat)%*%like_deriv)
}

newton.raphson <- function(x,h,y){
  theta = c(2,5,2) # sigma^2, eta, tau^2
  beta = 2
  count = 0
  while (count < 50){
    cq = calc.c(theta,x,length(y))
    beta = (1/(t(h)%*%cq[[2]]%*%h)*t(h)%*%cq[[2]]%*%y)[1]
    theta = theta - fish.score(cq[[2]],h,x,y,theta,beta,length(y))[,1]
    log_like = log.like(cq[[1]],cq[[2]],h,y,beta,length(y))[1]
    cat("\rbeta = ",beta,";  theta = [", theta,"];  log-likelihood = ", log_like)
    count = count +1
  }
  return(list(
    beta = beta,
    theta = theta
    ))
}
set.seed(1)
samps = create_samples(m = 200, sigma = 1, eta = 10, tau = 0.05, alpha = 1)
res = newton.raphson(samps$x,samps$h,samps$y)
  