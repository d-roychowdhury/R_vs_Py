args <- commandArgs(trailingOnly = TRUE)

## the seed
seed <- as.integer(args[1])

## dimesions n,p,r,u -------------------------------------------------------------------
n <- as.integer(args[2])
p <- as.integer(args[3])
r <- as.integer(args[4])
u <- as.integer(args[5])
NB <- as.integer(args[6])
#n_init_pts <- as.integer(args[7])
ncore <- as.integer(args[7])


#{{
 ## matrix inverse and square root functions--------------------------------------------
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
 
normvec <- function(a){
  return(sum(a^2))
  }
 
#}}---------------------------------------------------------------------------
# TRUE REGRESSION MODEL PARAMETERS
set.seed(1212)
eta_true <- matrix(runif(u*p,0,10),nrow=u,ncol=p)
A_true <- matrix(runif((r-u)*u,-1,1), nrow = r-u, ncol = u)
beta_true <- find_gammas_from_A(A_true)$gamma%*%eta_true ## calling find_gammas_from_A


mu_true <- runif(r,0,10)
l1 <- runif(u,0,10)
Omg_true <- diag(l1, u)
l2 <- runif(r-u,5,10)
Omg0_true <- diag(l2, r-u)

gamma_gamma0 <- find_gammas_from_A(A_true) ## calling find_gammas_from_A from functions.R
Gm_true <- gamma_gamma0$gamma
Gm0_true <- gamma_gamma0$gamma0
#---------------------------------------------------------------------------------------
set.seed(seed)

X <- matrix(rnorm(n*p),nrow=n,ncol=p) ## script X

mean_y <- mu_true+Gm_true%*%eta_true%*%t(X)
var_y <- Gm_true%*%Omg_true%*%t(Gm_true)+Gm0_true%*%Omg0_true%*%t(Gm0_true)

Y <- matrix(0,nrow=n,ncol=r)

for(j in 1:n){
  Y[j,] <- MASS::mvrnorm(1,mu = mean_y[,j],Sigma = var_y)
}

## Setting up the initial point based on freq estimate from Renvlp model:--------------

objects_to_export = c("X", "Y", "n", "p", "r", "u", "NB")
core_vec = c(1:ncore)
b_vec = c(1:NB)
dat = data.table::data.table(boot_idx = b_vec)
dat$core_assign = sample(core_vec, size = length(b_vec), replace = TRUE)
n_boot <- sapply(X=c(1:ncore),
                 FUN = function(x) return(sum(dat$core_assign == x)))
print(dat)
print(n_boot)
boot_se <- function (core_id) 
{
  print("the core_id")
  print(core_id)
  print("inside boot_se function")
  X <- as.matrix(X)
  a <- dim(Y)
  n <- a[1]
  r <- a[2]
  p <- ncol(X)

  fit <- senv(X, Y, u, asy = F)
  Yfit <- matrix(1, n, 1) %*% t(fit$mu) + X %*% t(fit$beta)
  res <- Y - Yfit
  print("ended res calc")
  bootenv <- function(i) {
    print(i)
    
    res.boot <- res[sample(1:n, n, replace = T), ]
    Y.boot <- Yfit + res.boot
    return(c(senv(X, Y.boot, u, asy = F)$beta))
  }
  bootbeta <- lapply(1:n_boot[core_id], function(i) bootenv(i))
  bootbeta <- matrix(unlist(bootbeta), nrow = n_boot[core_id], byrow = TRUE)
  return(bootbeta)
}
library(future.apply)
plan(multicore, workers = ncore)
bootbeta <- future_lapply(X=1:ncore,
                          FUN = function(core_id) boot_se(core_id),
                          future.globals = c(objects_to_export, "dat", "n_boot"),
                          future.packages = c("Renvlp"))

print("the bootbeta")
print(bootbeta)
#bootbeta <- unlist(bootbeta, recursive = F)
print("after bootbeta is unlisted")
bootbeta <- do.call(rbind, bootbeta) ## this is the bootstrap sample of beta: beta_boot_env, we want to compare this with beta_sample2 of or error_cal.R

beta_true_vec <- as.vector(beta_true)

err_boot_beta_env <- mean(sapply(X=c(1:NB), FUN= function(x) return(normvec(bootbeta[x,]-beta_true_vec)) ))

dim_beta_vec <- r*p
beta_boot_mean <- sapply(X =c(1:dim_beta_vec), FUN =function(x)  return(mean(bootbeta[,x])))
print(beta_boot_mean)
print(beta_true_vec)
err_freq_boot_beta_env <- normvec(beta_boot_mean - beta_true_vec)
object = list()
object$err_freq_boot_beta_env  <- err_freq_boot_beta_env
object$err_boot_beta_env <- err_boot_beta_env

print(object)

