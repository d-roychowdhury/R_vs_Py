## EXPERIMENT: Check how to use Benvlp_MC_resp_gibbs function from the Benvlp package ( especially Benvlp::Benvlp_MC_rsp_gibbs)

args <- commandArgs(trailingOnly = TRUE)

## the seed
seed <- as.integer(args[1])

## dimesions n,p,r,u -------------------------------------------------------------------
n <- as.integer(args[2])
p <- as.integer(args[3])
r <- as.integer(args[4])
u <- as.integer(args[5])

# List of required packages
packages <- c("Renvlp", "MASS","tcltk", "BAMBI", "Bessel")

# Load packages
invisible(lapply(packages, library, character.only = TRUE))

#### GENERATION OF X,Y ===========================================================
set.seed(1212) ## to fix true model param

##{{{ fns cholsove,sqrtmat, mat_sval_sqrt_inv, find_gammas_from_A, 
 
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

##}}}
##----------------------------------------------------------------------------------------
# TRUE REGRESSION MODEL PARAMETERS
eta_true <- matrix(runif(u*p,0,10),nrow=u,ncol=p)
A_true <- matrix(runif((r-u)*u,-1,1), nrow = r-u, ncol = u)
beta_true <- find_gammas_from_A(A_true)$gamma%*%eta_true ## calling find_gammas_from_A

#print(dim(beta_true))
mu_true <- runif(r,0,10)
l1 <- runif(u,0,10)
Omg_true <- diag(l1, u)
l2 <- runif(r-u,5,10)
Omg0_true <- diag(l2, r-u)

gamma_gamma0 <- find_gammas_from_A(A_true) ## calling find_gammas_from_A from functions.R
Gm_true <- gamma_gamma0$gamma
Gm0_true <- gamma_gamma0$gamma0

## data generation--------------------------------------------------------------

#Set the random seed for data
set.seed(seed)

X <- matrix(rnorm(n*p),nrow=n,ncol=p) ## script X

mean_y <- mu_true+Gm_true%*%eta_true%*%t(X)
var_y <-Gm_true%*%Omg_true%*%t(Gm_true)+Gm0_true%*%Omg0_true%*%t(Gm0_true)

Y <- matrix(0,nrow=n,ncol=r)

for(j in 1:n){
  Y[j,] <- MASS::mvrnorm(1,mu = mean_y[,j],Sigma = var_y)
}

## hyper params needed MAPCA.R---------------------------------------------------
Psi <- 0.001*diag(1,u)
Psi0 <- 0.001*diag(1,(r-u))
nu <- u 
nu0 <- r-u
K <- 1e3*diag(1,(r-u)) # calling I_cov and zero_mean from functions.R
L <- 1e3*diag(1,u)  # ditto
#M <- 
A0 <- matrix(0, nrow = (r-u), ncol =u)

detach("package:Renvlp", unload = TRUE) ## need to detach this so that Benvlp's own env and envMU which are different from renvlp packages can be used.


## define K.alf.inv and L.half.inv
#K.half.inv <- sqrtmatinv(K)
#L.half.inv <- sqrtmatinv(L)

library(Benvlp)
## applying the Benvlp_MC_resp_gibbs functions:=====================================
n_mcmc = 1e4
time_mcmc <- system.time({mcmc_res <- Benvlp_MC_resp_gibbs(X,Y,u, 
                      n.iter = 15000,
                      burnin.prop =1/3,
                      n.chains =1,
                      init = "mle")}) ## the default inint = "map" so we need to provide hyper parameters.
#print(attributes(mcmc_res))
print(time_mcmc[3]/60)
#### FINDING THE EFFECTIVE SAMPLE SIZE.=============================================
## ask Sapta-da how to do that.

# Step 1: Convert the list into a matrix
# Assuming mcmc_res$A$Chain_1 is the list
mcmc_beta_matrix <- do.call(rbind, lapply(mcmc_res$beta$Chain_1, function(x) as.vector(x)))

normvec <- function(a){
  return(sum(a^2))
  }
beta_sample_mcmc <- lapply(X=c(1:n_mcmc), FUN = function(x) return(matrix(mcmc_beta_matrix[x,], ncol =p, nrow = r)))

#print(dim(beta_sample_mcmc[[1]]))
#print(dim(beta_true))
  err_bayes_beta_mcmc <- mean(sapply(X=c(1:n_mcmc), FUN= function(x) return(normvec(beta_sample_mcmc[[x]]-beta_true)) ))

beta_freq <- Reduce('+',beta_sample_mcmc)/n_mcmc
err_freq_beta_mcmc <- norm((beta_freq-beta_true),type="2")
object <- list()
object$err_bayes_beta_mcmc = err_bayes_beta_mcmc
object$err_freq_beta_mcmc = err_freq_beta_mcmc
print(object)
