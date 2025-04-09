#---------------------------------------------------------
#library(expm, quietly = TRUE) # for sqrtm
#library(fastmatrix, quietly = TRUE) # for commutation matrix
library(matrixNormal, quietly = TRUE)# MN random generation +is.positive.definite function




{{{ #fns cholsove,sqrtmat, mat_sval_sqrt_inv, find_gammas_from_A, 
 
 
normvec <- function(a){
  return(sum(a^2))
  }
 
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

find_gammas_from_A <- function(A) {
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
  
  
 return(list("gamma" =gamma, "gamma0" =gamma0))
}

# FUNCTION TO GET A FROM GAMMA
get_A <- function(G){
  G1 <- G[c(1:u),]
  G2 <- G[c((u+1):r),]
  tmp <- G2%*%solve(G1)
  return(tmp)
}

}}}

##THE TIME ELAPSED+ERRORS-----------------------------------------------------------------
## function cw()

## make sure A_opt[[x]] for x =1:NB are the optimizers! A_opt=lst_A_opt_futlap_final

A_opt <- lst_A_opt_futlap_final 

 cw <- function(z,w){
       zbw <- colSums(diag(w)%*%z)/sum(w)
       tmp1 <- z-rep(1,n)%*%t(zbw)
       return(tmp1)
        }
 
## function anti_vec()
anti_vec <- function(x) {
  return(matrix(x,byrow=FALSE, nrow=r-u))
}

# Firstly generation of eta for each A_opt[[x]], x=1,...,NB
ev_gen <- function(){ ## used in eta_gen function defined next
  Yc <- cw(Y,rep(1,n))
  Xc <- cw(X,rep(1,n))
  t1_rep <- t(Xc)%*%Xc + M
  
  e_v <- (t(Yc)%*%Xc)%*%solve(t1_rep)
  
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

comparisons_wbb <- function(NB,A_arg){ # where t is the time taken to calculate A_arg
  
  
  time_common <- system.time({
  A_wbb<-0
  
  if(NB>1){
    A_wbb <- Reduce('+', A_arg)/NB
  } else if(NB == 1){
    A_wbb <- A_arg
  }
  })
  
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

   # sample error of beta1
  err_freq_beta1 <- norm((beta1-beta_true),type="2")
  

  # ESITMTION USING METHOD 2
  # eta_w drawn from it's conditional posterior MNup() and beta_w calculated for each wi's. (method2)
  
  time_beta2<- system.time({
  eta_NB2 <- lapply(X=c(1:NB),  FUN = function(x) return(eta_gen(x)))
  beta_sample2 <- lapply(X=c(1:NB), FUN = function(x) return(find_gammas_from_A(anti_vec(A_opt[[x]]))$gamma%*%eta_NB2[[x]])) 
  beta2 <- Reduce('+',beta_sample2)/NB 
  })
  

   # freq estimate error of beta2
  err_freq_beta2 <- norm((beta2-beta_true),type="2")
   # bayes estimate of error in beta2
  err_bayes_beta2 <- mean(sapply(X=c(1:NB), FUN= function(x) return(normvec(beta_sample2[[x]]-beta_true)) ))
   err_freq_beta_env <- normvec(beta_env - beta_true)
  
  return(list("NB" = NB,
              #"beta1" = beta1,
              #"beta2" = beta2,
              #"beta_env" = beta_env,
              #"time_beta1" = time_beta1[3]/60,
              #"time_beta2" = time_beta2[3]/60,
              "err_freq_beta1" = err_freq_beta1,
              "err_freq_beta2" = err_freq_beta2,
              "err_bayes_beta_cbb" = err_bayes_beta2,
              "err_freq_beta_env" = err_freq_beta_env))
  
}


##-----------------------------------------------------------------------------------------
### OBJECTS TO BE SAVED
#object1 <- list("beta_ols"=beta_ols,
#                "beta_env" = beta_env,
#                "err_ols" = err_ols,
#                "err_env" = err_env)


#object2 <-comparisons_wbb(100,A_opt, time)
# object3 <- mcmc/benvlp estimates of beta, time taken for estimation and errors!

#object <- append(object1,object2)

## THE OUTPUT
# object 
#saveRDS(object, file = paste0("output_", n, "_", iter_number,  ".rds")) 
