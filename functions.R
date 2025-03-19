###########-------------------------------------------------------------------------------------------


## matrix inverse and square root functions--------------------------------------
cholsolve <- function(M) { # for efficient inversion of pd matrices
  return(chol2inv(chol(M)))
}
## for mat^1/2
sqrtmat <- function(mat) {
  e <- eigen(mat, symmetric = TRUE)
  if (length(e$values) == 1) {
    sqrt(mat)
  } else {
    e$vectors %*% diag(c(sqrt(e$values))) %*% t(e$vectors)
  }
  # t(chol(mat))
}

## for mat^-1 and mat^-1/2
mat_svd_sqrt_inv <- function(mat) {
  e <- eigen(mat, symmetric = TRUE)
  d_mat <- dim(mat)
  d1 <- d_mat[1]
  d2 <- d_mat[2]
  Dmat_list <- list(
    # orig = diag(e$values, d1, d2),
    inv = diag(1 / e$values, d1, d2),
    # sqrt = diag(sqrt(e$values), d1, d2),
    inv_sqrt = diag(1 / sqrt(e$values), d1, d2)
  )
  out <- lapply(
    Dmat_list,
    function(this_D) {
      e$vectors %*% this_D %*% t(e$vectors)
    }
  )
  out
}


find_gammas_from_A <- function(A,
                               jacobian = FALSE,
                               log = TRUE,
                               jacobian_only_gamma = TRUE,
                               return_jacob_mat = FALSE) {
  dims <- dim(A)
  u <- dims[2]
  r <- sum(dims)
  
  CA <- matrix(0, nrow = r, ncol = u)
  CA[(u + 1):r, ] <- A
  CA[1:u, 1:u] <- diag(1, u)
  CAtCA <- crossprod(CA)
  svd_CAtCA <- mat_svd_sqrt_inv(CAtCA)
  CAtCA_minus_half <- svd_CAtCA$inv_sqrt
  CAtCA_inv <- svd_CAtCA$inv
  # CAtCA_half <- svd_CAtCA$sqrt
  gamma <- CA %*% CAtCA_minus_half
  
  
  DA <- matrix(0, nrow = r, ncol = r - u)
  DA[1:u, ] <- -t(A)
  DA[-(1:u), ] <- diag(1, r - u)
  DAtDA <- crossprod(DA)
  svd_DAtDA <- mat_svd_sqrt_inv(DAtDA)
  DAtDA_minus_half <- svd_DAtDA$inv_sqrt
  DAtDA_inv <- svd_DAtDA$inv
  # DAtDA_half <- svd_DAtDA$sqrt
  gamma0 <- DA %*% DAtDA_minus_half
  
  
  if (jacobian | return_jacob_mat) {
    Jmat_gamma_A <- jacobian_gamma_A(
      u = u, r = r, A = A,
      CA = CA, CAtCA = CAtCA,
      # CAtCA_half,
      CAtCA_inv = CAtCA_inv,
      CAtCA_minus_half = CAtCA_minus_half
    )
    
    
    if (jacobian_only_gamma) {
      # returns jacobian from only the gamma part
      Jmat_gamma0_A <- NULL
    } else {
      Jmat_gamma0_A <- jacobian_gamma0_A(
        u = u, r = r, A = A,
        DA = DA, DAtDA = DAtDA,
        # DAtDA_half,
        DAtDA_inv = DAtDA_inv,
        DAtDA_minus_half = DAtDA_minus_half
      )
    }
    
    if (jacobian) {
      Jmat_A <- rbind(Jmat_gamma_A, Jmat_gamma0_A)
      det_Jacobian_Omat_A <- mat_vol(Jmat_A, logarithm = log)
    } else {
      Jmat_A <- det_Jacobian_Omat_A <- NULL
    }
  } else {
    Jmat_gamma_A <- Jmat_gamma0_A <- det_Jacobian_Omat_A <- NULL
  }
  
  out <- list(
    gamma = gamma,
    gamma0 = gamma0,
    CA = CA, CAtCA = CAtCA,
    DA = DA, DAtDA = DAtDA,
    det_Jacobian_Omat_over_A = det_Jacobian_Omat_A
  )
  
  if (return_jacob_mat) {
    out$jacob_mat_gamma_over_A <- Jmat_gamma_A
    out$jacob_mat_gamma0_over_A <- Jmat_gamma0_A
  }
  
  out
}


##------------------------------------------------------------------------------------------------
# FUNCTION TO GET A FROM GAMMA
get_A <- function(G){
  G1 <- G[c(1:u),]
  G2 <- G[c((u+1):r),]
  tmp <- G2%*%solve(G1)
  return(tmp)
}

## functions CA and DA-----------------------------------------------------------------------------

CA <- function(A) {
  rbind(diag(u),A)
}
DA <- function(A) {
  rbind(-t(A),diag(r-u))
}

######-------------------------------------------------------------------------------------------------
## functions for generating zero means and identity covariance matr.
zero_mean <- function(d1,d2){
  return(matrix(0,nrow=d1,ncol=d2))
}
I_cov <- function(d){
  return(diag(1,d))
}








####----------------------------------------------------------------------------------------------

# firstly, functions needed to get values: Xcw,Ycw,Dw,Nw,Gwt, needed for log_post

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

# secondy - functions for d_Gam,d_gam0,d_CA,d2_CA,d_DA,d2_DA, needed for deriv of log_post

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

#----------------------------------------------------------------------------------------

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
  Mwt <- t(Xcw)%*%Dw%*%Xcw+M
  ewt <- t(solve(Mwt)%*%(t(Xcw)%*%Dw%*%Ycw+M%*%t(e)))
  gwt <- t(Ycw)%*%Dw%*%Ycw + e%*%M%*%t(e)-ewt%*%Mwt%*%t(ewt)
  
  return(gwt)
}



### defining the log-post and deriv of log-post functions to be used in foreach

# THE log weighted posterior function 

log_post_w <- function(A,i){## needs G1,Gwt
  
  nw <- Nw[[i]]
  c1 <- -(nu+nw-1)/2
  c2 <- -(nu0+nw-1)/2
  g_g0 <- find_gammas_from_A(A)
  Gm <- g_g0$gamma
  Gm0 <- g_g0$gamma0
  
  gwt <- Gwt[[i]]
  g1 <- G1[[i]]
  
  t1 <- t(Gm)%*%gwt%*%Gm + Psi
  t2 <- t(Gm0)%*%g1%*%Gm0 + Psi0
  tmp3 <- (A-A0)
  t3 <- K_inv%*%tmp3%*%L_inv%*%t(tmp3)
  
  ld_t1 <- determinant(t1, logarithm=TRUE)$modulus[1]
  ld_t2 <- determinant(t2, logarithm=TRUE)$modulus[1]
  
  # final <- c1*log(det(t1)) + c2*log(det(t2)) -1/2*tr(t3)
  final <- c1*ld_t1 + c2*ld_t2 -1/2*tr(t3)
  
  #print("log_post_w OK")
  return(final)
}

# THE d_lpda_1,d_lpda_2,d_lpda_3 FINALLY d_lpda func


d_lpda_1 <- function(A,i) { ## need Gwt
  
  
  nw <- Nw[[i]]
  c1 <- -(nu+nw-1)/2
  g_g0 <- find_gammas_from_A(A)
  Gm <- g_g0$gamma
 
  
  gwt <- Gwt[[i]]
  
  
  tmp11 <- t(Gm) %*% t(gwt) %*% Gm + t(Psi)
 # print(dim(tmp11)) # cehcker
  ter1 <- solve(tmp11)
  ter21 <- ((t(Gm) %*% t(gwt)) %x% diag(u)) %*% commutation(r,u,matrix=TRUE) ## commutation(r,u)
  ter22 <- diag(u) %x% (t(Gm) %*% gwt)
  ter3 <- d_Gam(A) 
  
  print("d_lpda_1ok") # checker
  return (c1*vec(ter1) %*% (ter21 + ter22) %*% ter3)
  
}
d_lpda_2 <- function(A,i) { ## needs G1
  
  nw <- Nw[[i]]
  
  c2 <- -(nu0+nw-1)/2
  g_g0 <- find_gammas_from_A(A)
  Gm0 <- g_g0$gamma0
 
  g1 <- G1[[i]]
  
  t2_1 <- t(Gm0) %*% g1 %*% Gm0+t(Psi0)
 # print(dim(t2_1)) #
 # eg_val <- eigen(t2_1)$values #
 # print(c(min(eg_val), max(eg_val))) #

  ter1 <- solve(t2_1)
  ter2_1 <- (t(Gm0) %*% t(g1) %x% diag(r-u)) %*% commutation(r,r-u, matrix=TRUE) ##commutation(r,r-u)
  ter2_2 <- diag(r-u) %x% (t(Gm0) %*% g1)
  ter3 <- d_Gam0(A)
  
  print("d_lpda_2ok")
  return(c2*vec(ter1) %*% (ter2_1 + ter2_2) %*% ter3) 
}

d_lpda_3 <- function(A) {
  
  tmp3 <- (A-A0)
  ter1 <- K_inv %*% tmp3 %*% L_inv
  ter2 <- t(K_inv) %*% tmp3 %*% t(L_inv)
  
  print("d_lpda_3ok")
  return(-0.5*vec((ter1+ter2)))
}
d_lpda <- function(A,i) {
  return(as.vector(d_lpda_1(A,i) + d_lpda_2(A,i) + d_lpda_3(A)))
}

### Expressing lpda,d_lpda as functions of vec(A)


f <- function(x,i){ #x is a (r-u)u*1 vector
  A_matrix <- matrix(x,byrow=FALSE, nrow=r-u)
  #print("f ok")
  return(log_post_w(A_matrix,i))
}

# function to be minimized: F_minus
F_minus <- function(x,i){
  return(-f(x,i))
}
# gradient function of F_minus
GR_minus <- function(x,i) {
  x_matrix = matrix(x, nrow=r-u)
  return (-d_lpda(x_matrix,i))
}





# compare errors related functions

# Firstly generation of eta for each A_opt[[x]], x=1,...,NB
ev_gen <- function(){ ## used in eta_gen function defined next
  Yc <- cw(Y,rep(1,n))
  Xc <- cw(X,rep(1,n))
  t1_rep <- t(Xc)%*%Xc + M
  t2_rep <- e%*%M
  
  e_v <- (t(Yc)%*%Xc+t2_rep)%*%solve(t1_rep)
  
  return(e_v)
} 

eta_gen <- function(x){
  g_g0 <- find_gammas_from_A(anti_vec(A_opt[[x]]))
  G_A <- g_g0$gamma

  
  e_v <- ev_gen()
  etaw_mean <- t(G_A)%*%e_v
  etaw_cov1 <- Omg_true
  Xc <- cw(X,rep(1,n))
  etaw_cov2 <- solve(M+t(Xc)%*%Xc)
  
  eta_w <- matrixNormal::rmatnorm(1,etaw_mean,etaw_cov1,etaw_cov2)
  
}


#comparisons should give beta1, beta2, err^freq_beta1, err^freq_beta2, err^bayes_beta2, comp_times of beta1, beta2

comparisons_wbb <- function(NB,A_arg,t){ # where t is the time taken to calculate A_arg
  
  
  time_common <- system.time({
  A_wbb<-0
  
  if(NB>1){
    A_wbb <- Reduce('+', A_arg)/NB
  } else if(NB == 1){
    A_wbb <- A_arg
  }
  })
  time_common <- time_common + t
  err_Aw <- norm((A_true-A_wbb), type="2")
  
  #print(anti_vec(A_wbb))
  
  
  # ESTIMATION USING METHOD 1
  
  time_beta1<- system.time({
    ev <- ev_gen()
  
  Gamma_Gamma0 <- find_gammas_from_A(anti_vec(A_wbb))
  Gamma_pe <- Gamma_Gamma0$gamma
  eta_w <- t(Gamma_pe)%*%ev
  beta1 <- Gamma_pe%*%eta_w
  })
  
  # time needed to esitmate beta1
   time_beta1 <- time_beta1 + time_common
   # sample error of beta1
  err_freq_beta1 <- norm((beta1-beta_true),type="2")
  

  # ESITMTION USING METHOD 2
  # eta_w drawn from it's conditional posterior MNup() and beta_w calculated for each wi's. (method2)
  
  time_beta2<- system.time({
  eta_NB2 <- lapply(X=c(1:NB),  FUN = function(x) return(eta_gen(x)))
  beta_sample2 <- lapply(X=c(1:NB), FUN = function(x) return(find_gammas_from_A(anti_vec(A_opt[[x]]))$gamma%*%eta_NB2[[x]])) 
  beta2 <- Reduce('+',beta_sample2)/NB 
  })
  
  # time needed to estimate beta2
  time_beta2 <- time_beta2 + time_common
   # freq estimate error of beta2
  err_freq_beta2 <- norm((beta2-beta_true),type="2")
   # bayes estimate of error in beta2
  err_bayes_beta2 <- mean(sapply(X=c(1:NB), FUN= function(x) return(normvec(beta_sample2[[x]]-beta2)) ))
   
  
  return(list("NB" = NB,
              "beta1" = beta1,
              "beta2" = beta2,
              "time_beta1" = time_beta1[3]/60,
              "time_beta2" = time_beta2[3]/60,
              "err_freq_beta1" = err_freq_beta1,
              "err_freq_beta2" = err_freq_beta2,
              "err_bayes_beta2" = err_bayes_beta2))
  
}


normvec <- function(a){
  return(sum(a^2))
}
