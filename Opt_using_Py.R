##EXPERIMENT: Find A_opt through Python:

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
#max_lbfgs_rep <- as.integer(args[7]) ## needed for Opt.py
#init_step_size <- as.numeric(args[8]) ## ditto
#hist_size <- as.integer(args[9])     ## ditto 
#numnber of cluster forparallelization(NOTNEEDEDHERE)-------------------------------------
#nclust <- as.integer(args[7])

## file_name is taken care of. instead we pass on
## the number of the simulation.
## file in which output is saved as R object
## file_name <- as.character(args[6])

# arguments are 0-based, we need 1-based file numbering
#iter_number <- as.integer(args[6]) + 1
#print(iter_number)

#list_of_settings <- list("n" =n, "p" =p, "r" =r, "u"=u, "nclust" =nclust, "n_simulations" =iter_number)
#print(list_of_settings)
#----------------------------------------------------------------------------------------
#list_of_settings <- list("n" =n, "p" =p, "r" =r, "u"=u, "NB" =NB, "max_lbfgs_rep" =max_lbfgs_rep)

list_of_settings <- list("n" =n, "p" =p, "r" =r, "u"=u, "NB" =NB)
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
print("dim of A_test on line 166")
print(dim(A_test))
#print(init_step_size)
#####------------------------------------------------------------------------------------------------

# generating w:
W <- matrix(rexp(n*NB), ncol = n)
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
py$X <- r_to_py(X)
py$Y <- r_to_py(Y)
py$W <- r_to_py(W)
py$nu <- r_to_py(nu)
py$nu0 <- r_to_py(nu0)
py$n <- r_to_py(n)
py$p <- r_to_py(p)
py$r <- r_to_py(r)
py$u <- r_to_py(u)
#py$max_lbfgs_rep <- r_to_py(max_lbfgs_rep) ## needed for Opt.Py
#py$init_step_size <- r_to_py(init_step_size) ## ditto
#py$hist_size <- r_to_py(hist_size)   ## ditto
py$NB <- r_to_py(NB)
py$A_test <- r_to_py(A_test)
py$A_env <- r_to_py(A_env)

# Convert list_of_settings to Python dictionary
py$list_of_settings <- r_to_py(list_of_settings)

##-----------------------------------------------------------------------------------------
## sourcing R for single initial point based optimization

##(O1) optimization using R
time_R_opt_with_one_init_pt <- system.time(
source("Opt.R"))[3]/60


##(D1) the deriv
#time_R_deriv <- system.time(
#source("R_based_deriv.R"))[3]/60


##-----------------------------------------------------------------------------------------
## sourcing python for single initial point based optimization

## (O1) lbfgs based  optimization
#time_Py_opt_with_one_init_pt <- system.time(
#source_python("Opt.py"))[3]/60

## (O2) minimize based optimization
#time_Py_opt_with_one_init_pt <- system.time(
#source_python("Opt_minimize.py"))[3]/60

## (O3) mscipy-minimize based optimization
time_scipy_minimize <- system.time(
source_python("Opt_w_wrapped_fun.py"))[3]/60

## (D1) the deriv
#time_Py_deriv <- system.time(
#source_python("Opt_minimize.py"))[3]/60

#time_Py_deriv <- system.time(
#source_python("Opt_w_wrapped_fun.py"))[3]/60
##----------------------------------------------------------------------------------------

## times for optimizations
#print(paste("time_R_opt_with_one_init_pt:", time_R_opt_with_one_init_pt))
#print(paste("time_Py_opt_with_one_init_pt:", time_Py_opt_with_one_init_pt))

## times for deriv computation
print(paste("time_R:", time_R_opt_with_one_init_pt))
print(paste("time_Py:",time_scipy_minimize))

