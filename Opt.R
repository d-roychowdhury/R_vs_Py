## EXPERIMENT: Optimize using R
A_init = A_env
#print(A_env)
# note W is NB * n

cw <- function(z,w){
  dw <- diag(w)
  Nw <- sum(w)
  zbw <- colSums(dw%*%z)/Nw
  tmp1 <- z-rep(1,n)%*%t(zbw)
  return(tmp1)
}

tr <- function(A){ # trace function needed in log_posterior 
  return(sum(diag(A)))
}

# functions needed for generating Nw, G1, Gwt 
Nwf <- function(w){
  return(sum(w)) 
}

G1f <- function(w){
  
  Dw <- diag(w)
  Ycw <- cw(Y,w)
  g1 <- t(Ycw)%*%Dw%*%Ycw
  
  return(g1)
}

Gwtf <- function(w){
  
  Dw <- diag(w)
  
  Xcw <- cw(X,w)
  Ycw <- cw(Y,w)
  Mwt <- t(Xcw)%*%Dw%*%Xcw
  ewt <- t(solve(Mwt)%*%(t(Xcw)%*%Dw%*%Ycw))
  gwt <- t(Ycw)%*%Dw%*%Ycw -ewt%*%Mwt%*%t(ewt)
  
  return(gwt)
}

Nw <- lapply(X=c(1:NB), function(x) return(Nwf(W[x,])))
G1 <- lapply(X=c(1:NB), function(x) return(G1f(W[x,])))
Gwt <- lapply(X=c(1:NB), function(x) return(Gwtf(W[x,])))


for( i in 1:NB){


  
  nw <- Nw[[i]]
  c1 <- -(nu+nw-1)/2
  c2 <- -(nu0+nw-1)/2

  gwt <- Gwt[[i]]
  g1 <- G1[[i]]
  
  
  ### defining the log_post
lpda <- function(A){## needs G1,Gwt
  
  g_g0 <- find_gammas_from_A(A)
  Gm <- g_g0$gamma
  Gm0 <- g_g0$gamma0
  
  t1 <- t(Gm)%*%gwt%*%Gm + Psi
  t2 <- t(Gm0)%*%g1%*%Gm0 + Psi0
  tmp3 <- (A-A0)
  t3 <- K_inv%*%tmp3%*%L_inv%*%t(tmp3)
  
  final <- c1*log(det(t1)) + c2*log(det(t2)) -1/2*tr(t3)
 
  return(final)
}
 ### defining d_lpda
 
d_lpda <- function(A){

  g_g0 <- find_gammas_from_A(A)
  Gm <- g_g0$gamma
  Gm0 <- g_g0$gamma0
  
##{{ FUNCTIONS NEEDED TO DEFINE d_lpda

CA <- function(A) {
  rbind(diag(u),A)
}
DA <- function(A) {
  rbind(-t(A),diag(r-u))
}


d_CA <- function(){ ## der. of Ca. w.r.t A 
  Mu <- rbind(matrix(0,nrow=u,ncol=r-u),diag(r-u)) ## check1
  return(bdiag(replicate(u,Mu,simplify=FALSE)))
} # indep of A

d_DA <- function(){ ## der of DA(A), w.r.t A
  t1 <- commutation(r-u,r,matrix= TRUE)
  t2 <- rbind(-diag(u*(r-u)),matrix(0,nrow=(r-u)^2,ncol=u*(r-u))) ## check2
  return(t1 %*% t2)
} # indep of A

d2_CA <- function(A){ ## der of (Ca'Ca)^-1/2 w.r.t Ca
   CA <-CA(A)
   CAtCA <- crossprod(CA)
  svd_CAtCA <- mat_svd_sqrt_inv(CAtCA)
  CAtCA_inv <- svd_CAtCA$inv
  CAtCA_minus_half <- svd_CAtCA$inv_sqrt
  
  #sqrt_tmp <- sqrtmat(tmp)
  t1 <-  CAtCA_minus_half%x% diag(u) + diag(u) %x% CAtCA_minus_half
  t1 <- -solve(t1)
  t2 <- CAtCA_inv%x% (CAtCA_inv%*%t(CA))
  t3 <- (CAtCA_inv%*%t(CA)) %x% CAtCA_inv
  return(t1 %*% (t2 + t3%*%commutation(r,u,matrix=TRUE)))
}
d2_DA <- function(A){ ## der of (Da'Da)^-1/2 w.r.t Da
   
   DA <- DA(A)
   DAtDA <- crossprod(DA)
  svd_DAtDA <- mat_svd_sqrt_inv(DAtDA)
  DAtDA_inv <- svd_DAtDA $inv
  DAtDA_minus_half <- svd_DAtDA$inv_sqrt
  
  t1 <- DAtDA_minus_half %x% diag(r-u) + diag(r-u) %x% DAtDA_minus_half
  t1 <- -solve(t1)
  t2 <- DAtDA_inv %x% (DAtDA_inv%*%t(DA))
  t3 <- (DAtDA_inv%*%t(DA)) %x% DAtDA_inv
  return(t1 %*% (t2 + t3%*%commutation(r,r-u,matrix =TRUE)))  
}

d_Gam<- function(A){
  
  CA <- CA(A)
  t1 <- (diag(u) %x% CA) %*% d2_CA(A)
  
  
  CAtCA <- crossprod(CA)
  svd_CAtCA <- mat_svd_sqrt_inv(CAtCA)
  CAtCA_minus_half <- svd_CAtCA$inv_sqrt
  

  t2 <- CAtCA_minus_half %x% diag(r)
  return((t1+t2) %*% d_CA())
}
d_Gam0 <- function(A){
  
  DA <- DA(A)
  t1 <- (diag(r-u) %x% DA) %*% d2_DA(A)
  
  DAtDA <- crossprod(DA)
  svd_DAtDA <- mat_svd_sqrt_inv(DAtDA)
  DAtDA_minus_half <- svd_DAtDA$inv_sqrt
  

  t2 <- DAtDA_minus_half  %x% diag(r)
  return((t1+t2) %*% d_DA())
}


anti_vec <- function(x) {
  return(matrix(x,byrow=FALSE, nrow=r-u))
}

##}}
##### function for term 1 of deriv
d_lpda_1 <- function(A) { 
  
  tmp_mult1 = t(Gm)%*% t(gwt)  # same as t(Gm)%*%gwt as gwt is symm since M is symm.
  tmp11 <- tmp_mult1 %*% Gm  +t(Psi)
  ter1 <- solve(tmp11)
  
  ter21 <- (tmp_mult1 %x% diag(u)) %*% commutation(r,u,matrix=TRUE) ## commutation(r,u)
  ter22 <- diag(u) %x% (tmp_mult1)
  ter3 <- d_Gam(A) 
  
  return (c1*vec(ter1) %*% (ter21 + ter22) %*% ter3)
  }

# function for term 2 of deriv
d_lpda_2 <- function(A) { 
  
  tmp_mult2 <- t(Gm0) %*% g1 # same as t(Gm)%*%t(g1) since g1 is symm
  t2_1 <- tmp_mult2 %*% Gm0 +t(Psi0)
  ter1 <- solve(t2_1)
  
  ter2_1 <- (tmp_mult2 %x% diag(r-u)) %*% commutation(r,r-u, matrix=TRUE) ##commutation(r,r-u)
  ter2_2 <- diag(r-u) %x% (tmp_mult2)
  ter3 <- d_Gam0(A) 
  return(c2*vec(ter1) %*% (ter2_1 + ter2_2) %*% ter3) 
}

# function for term 3 of deriv
d_lpda_3 <- function(A) { # doesnt depend on i
  
  tmp3 <- (A-A0)
  ter1 <- K_inv %*% tmp3 %*% L_inv
  ter2 <- t(K_inv) %*% tmp3 %*% t(L_inv)

  return(-0.5*vec((ter1+ter2)))
}

  return(as.vector(d_lpda_1(A) + d_lpda_2(A) + d_lpda_3(A)))
}


### Expressing lpda,d_lpda as functions of vec(A)


f <- function(x){ #x is a (r-u)u*1 vector
  A_matrix <- matrix(x,byrow=FALSE, nrow=r-u)
  return(lpda(A_matrix))
}

# function to be minimized: F_minus
F_minus <- function(x){ ## returns a scalar
  return(-f(x))
}
# gradient function of F_minus
GR_minus <- function(x) {  ## returns a vector!
  x_matrix = matrix(x, nrow=r-u)
  return (-d_lpda(x_matrix))
}



result = optim(par = A_init, 
                fn = function(x) return(F_minus(x)),
                gr = function(x) return(GR_minus(x)), 
                method = "BFGS",
                control = list(maxit = 1000, trace = 1))
print(paste("Number of iterations:", result$counts[2]))
A_opt <- result$par
print(A_opt)
#print(paste("is A_opt double? :", typeof(A_opt)))
f_val <- f(A_opt)       
print(f_val)

}
