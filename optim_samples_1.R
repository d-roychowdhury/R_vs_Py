##EXPERIMENT: check if function values are correct in python.

time_1 <- system.time({
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

#----------------------------------------------------------------------------------------
list_of_settings <- list("n" =n, "p" =p, "r" =r, "u"=u, "NB" =NB, "ncore" = ncore)
print(list_of_settings)

# Packages needed:----------------------------------------------------------------------
# List of required packages
packages <- c("expm", "fastmatrix", "matrixNormal", "reticulate", "fastmatrix")

# Load packages
invisible(lapply(packages, library, character.only = TRUE))



#---------------------------------------------------------------------------------------




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
set.seed(1212) ## to fix true model param

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
A0 <- matrix(0, nrow = (r-u), ncol =u)
#e <- matrix(0, ncol = r, nrow = p)
M <- 1e-3*diag(1,p) # ditto
K_inv <- 1e-3*diag(1,(r-u)) ## this inverse works + calling cholsolve from functions.R
L_inv <- 1e-3*diag(1,u)  ## this inverse works
dim_of_A = (r-u)*u
A_test <- matrix(runif(dim_of_A,0,10), ncol = u)
print("dim of A_test on line 166")
#print(dim(A_test))
#####------------------------------------------------------------------------------------------------

# generating w:
W <- matrix(rexp(n*NB), ncol = n)
for(i in 1:NB)
    W[i,] <- W[i,]/sum(W[i,])
    
# generating the initial pts;
rminu = r-u
init_pts = matrix(0, ncol =rminu*u, nrow =n_init_pts)
for(i in 1:n_init_pts)   init_pts[i,] = runif(rminu*u,-10,10)
##-----------------------------------------------------------------------
library(Renvlp)
envlp <- Renvlp::env(X, Y, u)
beta_env <- envlp$beta
err_env <- norm(beta_env-beta_true, type="2") ## SECOND FREQ ESTIMATE
G_env <- envlp$Gamma
A_env <- get_A(G_env)

g1_gwt_optimized <- function(X, Y, w){
    
    Nw = sum(w)
    xw = X*w
    #xwbar = colSums(xw)
    #sum1 = sum(w)
    xwbar = colSums(xw)/Nw
    #xwbar = colSums(xw)
    xcw = scale(X, center = xwbar, scale = FALSE)
    #xcw.t_dw_xcw = crossprod((xw*w), xw)
    xcw_w = xcw*w
    
    Mwt = crossprod(xcw_w, xcw) + M
    #xcw.t_dw_xcw = crossprod((xcw*w), xcw)
    #Mwt = xcw.t_dw_xcw + M
    
    
    yw = Y*w
    #ywbar = colSums(yw)
    ywbar = colSums(yw)/Nw
    #ywbar = colSums(yw)
    ycw = scale(Y, center = ywbar, scale = FALSE)
    ycw.t_dw_ycw =  crossprod((ycw*w), ycw)
    
    xcw.t_dw_ycw = crossprod(xcw_w, ycw) ## delete later
    ewt = t(solve(Mwt) %*% crossprod(xcw_w, ycw))
    
    #G1 = ycw.t_dw_ycw/sum1
    G1 = ycw.t_dw_ycw
    #GWT = (ycw.t_dw_ycw - ewt %*% Mwt %*% t(ewt))/sum1
    GWT = (ycw.t_dw_ycw - ewt %*% Mwt %*% t(ewt))
    
    #return(list("G1" = G1, "GWT" = GWT, "ewt" =ewt, "xdy" = xcw.t_dw_ycw))
    return(list("G1" = G1, "GWT" = GWT, "NW" =Nw))

}

# define the function in R:
##-------------------------------------------------- the function with g1, gwt, nw within the function---------------------------------------------------
ith_boot_function_1<- function(i){
    w = W[i,]
    g1gwtnw <- g1_gwt_optimized(X,Y,w)
    g1 <- g1gwtnw$G1
    gwt <- g1gwtnw$GWT 
    nw <- g1gwtnw$NW
       feval_R <- function(A){
        
        A <- matrix(A, byrow = FALSE, nrow = r-u)
        g_g0 <- find_gammas_from_A(A)
        Gm <- g_g0$gamma
        Gm0 <- g_g0$gamma0
        
         ## constants common to lpda, d_lpda
         c1 <- -(nu+nw-1)/2
         c2 <- -(nu0+nw-1)/2
         
       t1 <- t(Gm)%*%gwt%*%Gm + Psi
       t2 <- t(Gm0)%*%g1%*%Gm0 + Psi0
       tmp3 <- (A_env-A0)
       t3 <- K_inv%*%tmp3%*%L_inv%*%t(tmp3)

       final <- c1*log(det(t1)) + c2*log(det(t2)) -1/2*tr(t3)
       return(-final)
       }
    return(feval_R)
}


A_min <- function(i){ 
  
  neg_lpda <- ith_boot_function_1(i)
  for(iter in 1:n_init_pts){
 
  if (iter==1){
      A_init <- A_env 
      result = optim(par = A_init, 
                     fn = function(x) return(neg_lpda(x)),
                     method = "BFGS")
      A_opt <- result$par
      f_opt <- neg_lpda(A_opt)
    } else if(iter >1){
    
       A_init <- init_pts[(iter-1),]
       result = optim(par = A_init, 
                      fn = function(x) return(neg_lpda(x)),
                      method = "BFGS")
      A_min <- result$par
      f_min <- neg_lpda(A_min)
      if(f_min < f_opt){
         A_opt <- A_min
         f_opt <- f_min 
         }
    }
   }
return(A_opt)
}

## the optimization
library(future.apply)
objects_to_export <- c("Psi", "Psi0", "K_inv", "L_inv", "M", "A0", "A_true", "nu", "nu0", "n", "p", "r", "u", "A_test", "A_env", "init_pts", "NB",
                       "W","X", "Y", "g1_gwt_optimized", "ith_boot_function_1", "A_min")
library(future.apply)
core_vec = c(1:ncore)
b_vec = c(1:NB)
dat = data.table::data.table(boot_idx = b_vec)
dat$core_assign = sample(core_vec, size = length(b_vec), replace = TRUE)

run_boot <- function(core_id){
  subset = dat$boot_idx[which(dat$core_assign == core_id)]


  A_opt <- lapply(X = subset, FUN = function(x) {
                                    print(x)
                                    return(A_min(x))
                                    })
 return(A_opt)
 }
})[3]/60 ## end of time_1
## planning the core distribution 
plan(multicore, workers = ncore)
time_par<- system.time({
  lst_A_opt_parallel <- future_lapply(
    X = 1:ncore,
    FUN = function(core_id) run_boot(core_id),
    future.globals = c(objects_to_export, "dat"),  
    future.packages = c("reticulate", "data.table"),
    future.seed = TRUE
  )
lst_A_opt_futlap_final <- unlist(lst_A_opt_parallel, recursive = FALSE)
})[3]/60 ## end of time_par

##post processing
time_2<- system.time({

# Convert each vector to a string (to compare vectors as whole units)
vector_strings <- sapply(lst_A_opt_futlap_final, function(v) paste(v, collapse = ","))
# Find unique vectors and count them
num_unique_vectors <- length(unique(vector_strings))
print("the number of unique A_opt^ws")
print(num_unique_vectors)

## the measures
A_opt <- lst_A_opt_futlap_final 
print(A_opt)
print(A_env)

## f-vals should be deleted eventually.
f_vals <- sapply(X=c(1:NB), FUN = function(x){
                                  neg_lpda <- ith_boot_function_1(x)
                                  return(neg_lpda(A_opt[[x]]))          
                                            })
print(f_vals)                                           
source("error_cal.R")
})[3]/60 # end of time_2
object <- comparisons_wbb(NB,A_opt)
tot_time <- time_1+time_par+time_2
object$tot_time <- tot_time
print(object)


                 


