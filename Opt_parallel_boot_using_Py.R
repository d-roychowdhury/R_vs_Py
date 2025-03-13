##EXPERIMENT: Find A_opt for parallelized bootstrap blocks through Python:

## code to read the arguments from the .sh file
args <- commandArgs(trailingOnly = TRUE)

## the seed
seed <- as.integer(args[1])

## dimesions n,p,r,u -------------------------------------------------------------------
n <- as.integer(args[2])
p <- as.integer(args[3])
r <- as.integer(args[4])
u <- as.integer(args[5])
NB <- as.integer(args[6])
n_init_pts <- as.integer(args[7])



# arguments are 0-based, we need 1-based file numbering
#iter_number <- as.integer(args[6]) + 1
#print(iter_number)

#list_of_settings <- list("n" =n, "p" =p, "r" =r, "u"=u, "nclust" =nclust, "n_simulations" =iter_number)
#print(list_of_settings)
#----------------------------------------------------------------------------------------
#list_of_settings <- list("n" =n, "p" =p, "r" =r, "u"=u, "NB" =NB, "max_lbfgs_rep" =max_lbfgs_rep)

list_of_settings <- list("n" =n, "p" =p, "r" =r, "u"=u, "NB" =NB, "n_init_pts" = n_init_pts)
print(list_of_settings)

# Packages needed:----------------------------------------------------------------------
# List of required packages
packages <- c("expm", "fastmatrix", "matrixNormal", "reticulate", "fastmatrix")

# Load packages
invisible(lapply(packages, library, character.only = TRUE))



#---------------------------------------------------------------------------------------


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


mu_true <- runif(r,0,10)
l1 <- runif(u,0,10)
Omg_true <- diag(l1, u)
l2 <- runif(r-u,5,10)
Omg0_true <- diag(l2, r-u)

gamma_gamma0 <- find_gammas_from_A(A_true) ## calling find_gammas_from_A from functions.R
Gm_true <- gamma_gamma0$gamma
Gm0_true <- gamma_gamma0$gamma0

## data generation-----------------------------------------------------------------------------------

#Set the random seed for data
set.seed(seed)

X <- matrix(rnorm(n*p),nrow=n,ncol=p) ## script X

mean_y <- mu_true+Gm_true%*%eta_true%*%t(X)
var_y <-Gm_true%*%Omg_true%*%t(Gm_true)+Gm0_true%*%Omg0_true%*%t(Gm0_true)

Y <- matrix(0,nrow=n,ncol=r)

for(j in 1:n){
  Y[j,] <- MASS::mvrnorm(1,mu = mean_y[,j],Sigma = var_y)
}

###---------------------------------------------------------------------------------------
# hyper-params (fixed, seed doesn't affect this)

Psi <- 0.001*diag(1,u)
Psi0 <- 0.001*diag(1,(r-u))
nu <- u 
nu0 <- r-u
K <- 1e3*diag(1,(r-u)) # calling I_cov and zero_mean from functions.R
L <- 1e3*diag(1,u)  # ditto
#M <- 
A0 <- matrix(0, nrow = (r-u), ncol =u)
#e <- matrix(0, ncol = r, nrow = p)
M <- 1e-3*diag(1,p) # ditto
K_inv <- 1e-3*diag(1,(r-u)) ## this inverse works + calling cholsolve from functions.R
L_inv <- 1e-3*diag(1,u)  ## this inverse works
dim_of_A = (r-u)*u
A_test <- matrix(runif(dim_of_A,0,10), ncol = u)

#####------------------------------------------------------------------------------------------------


# generating w:
W <- matrix(rexp(n*NB), ncol = n)
####-------------------------------------------------------------------------------------
# generating the initial pts;
rminu = r-u
init_pts = matrix(0, ncol =rminu*u, nrow =n_init_pts)
for(i in 1:n_init_pts)   init_pts[i,] = runif(rminu*u,0,10)

##---------------------------------------------------------------------------------------

## Setting up the initial point based on freq estimate from Renvlp model:
library(Renvlp)
envlp <- Renvlp::env(X, Y, u)
beta_env <- envlp$beta
err_env <- norm(beta_env-beta_true, type="2") ## SECOND FREQ ESTIMATE
G_env <- envlp$Gamma
A_env <- get_A(G_env)
#print(A_env)

###--------------------------------------------------------------------------------------
## TO SEND Psi, Psi_0, K_inv, L_inv, M, A0, X, Y, W into Python
# Convert R objects to Python
py$Psi <- r_to_py(Psi)
py$Psi0 <- r_to_py(Psi0)
py$K_inv <- r_to_py(K_inv)
py$L_inv <- r_to_py(L_inv)
py$M <- r_to_py(M)
py$A0 <- r_to_py(A0)
py$A_true = r_to_py(A_true)
py$X <- r_to_py(X)
py$Y <- r_to_py(Y)
py$W <- r_to_py(W) ## if i were to send only wi to python_wi, then this should be handled.
py$nu <- r_to_py(nu)
py$nu0 <- r_to_py(nu0)
py$n <- r_to_py(n)
py$p <- r_to_py(p)
py$r <- r_to_py(r)
py$u <- r_to_py(u)
py$NB <- r_to_py(NB) ## if i were to parallelize in R, this need not be sent to python_wi
py$A_test <- r_to_py(A_test)
py$A_env <- r_to_py(A_env)
py$init_pts <- r_to_py(init_pts)
# Convert list_of_settings to Python dictionary
py$list_of_settings <- r_to_py(list_of_settings)


##-----------------------------------------------------------------------------------------
## sourcing python for single initial point based optimization


## (O3) mscipy-minimize based optimization -- THIS WORKS FINE.
time_scipy_minimize <- system.time(
source_python("Opt_parallel_boot.py"))[3]/60

print(time_scipy_minimize)




