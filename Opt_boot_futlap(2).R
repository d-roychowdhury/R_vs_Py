##EXPERIMENT: Find A_opt for parallelized bootstrap blocks through Python -(parallel in R)

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
ncore <- as.integer(args[8])


# arguments are 0-based, we need 1-based file numbering
#iter_number <- as.integer(args[6]) + 1
#print(iter_number)

#list_of_settings <- list("n" =n, "p" =p, "r" =r, "u"=u, "nclust" =nclust, "n_simulations" =iter_number)
#print(list_of_settings)
#----------------------------------------------------------------------------------------
#list_of_settings <- list("n" =n, "p" =p, "r" =r, "u"=u, "NB" =NB, "max_lbfgs_rep" =max_lbfgs_rep)

list_of_settings <- list("n" =n, "p" =p, "r" =r, "u"=u, "NB" =NB, "n_init_pts" = n_init_pts, "ncore" = ncore)
print(list_of_settings)

# Packages needed:----------------------------------------------------------------------
# List of required packages
packages <- c("expm", "fastmatrix", "matrixNormal", "reticulate", "future.apply", "data.table", "Renvlp", "MASS")

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

g1_gwt_optimized <- function(X, Y, w){
    
    xw = X*w
    #xwbar = colSums(xw)
    sum1 = sum(w)
    xwbar = colSums(xw)/sum1
    xcw = scale(X, center = xwbar, scale = FALSE)
    #xcw.t_dw_xcw = crossprod((xw*w), xw)
    xcw_w = xcw*w
    
    Mwt = crossprod(xcw_w, xcw) + M
    #xcw.t_dw_xcw = crossprod((xcw*w), xcw)
    #Mwt = xcw.t_dw_xcw + M
    
    
    yw = Y*w
    #ywbar = colSums(yw)
    ywbar = colSums(yw)/sum1
    ycw = scale(Y, center = ywbar, scale = FALSE)
    ycw.t_dw_ycw =  crossprod((ycw*w), ycw)
    
    xcw.t_dw_ycw = crossprod(xcw_w, ycw) ## delete later
    ewt = t(solve(Mwt) %*% crossprod(xcw_w, ycw))
    
    G1 = ycw.t_dw_ycw/sum1
    GWT = (ycw.t_dw_ycw - ewt %*% Mwt %*% t(ewt))/sum1
    
    #return(list("G1" = G1, "GWT" = GWT, "ewt" =ewt, "xdy" = xcw.t_dw_ycw))
    return(list("G1" = G1, "GWT" = GWT))

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

# generating w: where W is NB x n!
W <- matrix(rexp(n*NB), ncol = n)
#for(i in 1:NB)
#    W[i,] <- W[i,]/sum(W[i,])

# generating the initial pts;
rminu = r-u
init_pts = matrix(0, ncol =rminu*u, nrow =n_init_pts)
for(i in 1:n_init_pts)   init_pts[i,] = runif(rminu*u,0,10)
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


# calculating the G1 and GWT's for all w in W
g1_gwt_mat <- lapply(
                     X = 1:NB,
                     FUN = function(boot_num) g1_gwt_optimized(X,Y,W[boot_num,])
                     )
g1_mat <- lapply(g1_gwt_mat, function(entry) entry$G1)
print("------------------------------------in R ------------------------------------")
print(g1_mat)
print("----------------------------------- in R --------------------------------------")
gwt_mat <- lapply(g1_gwt_mat, function(entry) entry$GWT)
##---------------------------------------------------------------------------------------

## Setting up the initial point based on freq estimate from Renvlp model:
library(Renvlp)
envlp <- Renvlp::env(X, Y, u)
beta_env <- envlp$beta
err_env <- norm(beta_env-beta_true, type="2") ## SECOND FREQ ESTIMATE
G_env <- envlp$Gamma
A_env <- get_A(G_env)
#print(A_env)


##-----------------------------------------------------------------------------------------
## sourcing python file for "one boot block" operation,  & running them in parallel from R.
library(reticulate)
library(parallel)


# parallelizing using future_lapply:---------------------------------------------
#ncore = parallel::detectCores()-2
#ncore = 4
core_vec = c(1:ncore)
b_vec = c(1:NB)
dat = data.table::data.table(boot_idx = b_vec)
dat$core_assign = sample(core_vec, size = length(b_vec), replace = TRUE)
print(dat)
## common objects to export to python:
#objects_to_export <- c("Psi", "Psi0", "K_inv", "L_inv", "M", "A0", "A_true", "X", "Y", "nu", "nu0", "n", "p", "r", "u", "A_test", "A_env", "init_pts", "NB","W", "g1_mat", "gwt_mat")

## no need to export X,Y, instead we should export g1_mat and gwt_mat
objects_to_export <- c("Psi", "Psi0", "K_inv", "L_inv", "M", "A0", "A_true", "nu", "nu0", "n", "p", "r", "u", "A_test", "A_env", "init_pts", "NB","W", "g1_mat", "gwt_mat")

run_boot <- function(core_id){
## when we need to use run_boot within future_lapply, we need to export objects from the global environment to the parallel workers
  ## the subset of W that is to be sent to a specific core.
  subset = dat$boot_idx[which(dat$core_assign == core_id)]
  w_subset <- W[c(subset),] 
  g1_submat<- g1_mat[c(subset)]
  gwt_submat<-  gwt_mat[c(subset)]
  
  dims <- unique(lapply(g1_submat, dim))
  dim_vec <- c(dims[[1]], length(g1_submat))
  
  ## converting g1_submat and gwt_submat in to proper forms so they are accessible in python
  g1_submat_arr <- array(unlist(g1_submat), dim_vec)
  g1_submat <- aperm(g1_submat_arr, perm =c(3,1,2))
  
  gwt_submat_arr <- array(unlist(gwt_submat), dim_vec)
  gwt_submat <- aperm(gwt_submat_arr, perm =c(3,1,2))
  

  py$Psi <- r_to_py(as.matrix(Psi))
  py$Psi0 <- r_to_py(as.matrix(Psi0))
  py$K_inv <- r_to_py(as.matrix(K_inv))
  py$L_inv <- r_to_py(as.matrix(L_inv))
  py$M <- r_to_py(as.matrix(M))
  py$A0 <- r_to_py(as.matrix(A0))
  py$A_true <- r_to_py(as.matrix(A_true))
  #py$X <- r_to_py(as.matrix(X))
  #py$Y <- r_to_py(as.matrix(Y))
  py$nu <- r_to_py(nu)
  py$nu0 <- r_to_py(nu0)
  py$n <- r_to_py(n)
  py$p <- r_to_py(p)
  py$r <- r_to_py(r)
  py$u <- r_to_py(u)
  py$A_test <- r_to_py(as.matrix(A_test))
  py$A_env <- r_to_py(as.matrix(A_env))
  py$init_pts <- r_to_py(as.matrix(init_pts))
  py$NB <- r_to_py(NB)
  py$w_subset <- r_to_py(as.matrix(w_subset))
  py$g1_submat <- r_to_py(g1_submat)
  py$gwt_submat <- r_to_py(gwt_submat)
 # run the core-based-optimization-python file.
  py_env <- py_run_file("Opt_boot_block_core_id(3).py") ## change the python code name here as needed.
  A_opt_list = py_env$A_opt_results
 #print(A_opt_list)
  return(A_opt_list)
}
#time_series <- system.time({
#lst_A_opt_series = lapply(1:ncore, function(core_id) run_boot(core_id))})[3]/60
#print(lst_A_opt)
#lst_A_opt_s_final = unlist(lst_A_opt_series, recursive = FALSE)


# Set up multicore plan
plan(multicore, workers = ncore)
# run sequentially
#plan(sequential)
# Use future_lapply with multicore
time_futlap <- system.time({
  lst_A_opt_parallel <- future_lapply(
    X = 1:ncore,
    FUN = function(core_id) run_boot(core_id),
    future.globals = c(objects_to_export, "dat"),  
    future.packages = c("reticulate", "data.table"),
    future.seed = TRUE
  )
})[3] / 60
lst_A_opt_futlap_final <- unlist(lst_A_opt_parallel, recursive = FALSE)

## check if the output A_opt's are similar for series and parallel
#print(lst_A_opt_s_final[[1]])
print(lst_A_opt_futlap_final[[1]])
print(A_true)
## check how the runtimes differ
#print(time_series)
print(time_futlap)


