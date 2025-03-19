## code to read the arguments from the .sh file
args <- commandArgs(trailingOnly = TRUE)

## the seed
seed <- as.integer(args[1])

## dimesions n,p,r,u
n <- as.integer(args[2])
p <- as.integer(args[3])
r <- as.integer(args[4])
u <- as.integer(args[5])

## file_name is taken care of. instead we pass on
## the number of the simulation.
## file in which output is saved as R object
## file_name <- as.character(args[6])

# arguments are 0-based, we need 1-based file numbering
iter_number <- as.integer(args[6]) + 1
print(iter_number)

# numnber of cluster for parallelization
nclust <- as.integer(args[7])

list_of_settings <- list("n" =n, "p" =p, "r" =r, "u"=u, "nclust" =nclust, "n_simulations" =iter_number)
print(list_of_settings)
## {{{ first part of the code
## #---------------------------------------------------------
## # (DELETE FIRST 5 LINES ON PAPON WHEN USING .SH FILE)
## seed <- set.seed(12)
## n <- 50
## p <- 7
## r <- 20
## u <- 2

#---------------------------------------------------------
library(expm, quietly = TRUE) # for sqrtm
library(fastmatrix, quietly = TRUE) # for commutation matrix
library(matrixNormal, quietly = TRUE)# MN random generation +is.positive.definite function



set.seed(1212) ## to fix true model param

# SOURCE THE FUNCTIONS.R FILE HERE:
# Set the working directory to where your scripts are located (if not already set)
#setwd("/home/your_user/my_project/") # -------- WHAT'S THIS IN MY CASE ON PAPON


# Source functions.R from the same directory
source("./functions.R")
## what functions need to be exported to the foreach loop
all_objects <- ls(envir = .GlobalEnv)

# TRUE REGRESSION MODEL PARAMETERS
eta_true <- matrix(runif(u*p,0,10),nrow=u,ncol=p)
A_true <- matrix(runif((r-u)*u,-1,1), nrow = r-u, ncol = u)
beta_true <- find_gammas_from_A(A_true)$gamma%*%eta_true ## calling find_gammas_from_A from functions.R


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
####-------------------------------------------------------------------------------------------------
# hyper-params (fixed, seed doesn't affect this)

Psi <- 0.001*I_cov(u)
Psi0 <- 0.001*I_cov(r-u)
nu <- u 
nu0 <- r-u
K <- 1e3*I_cov(r-u) # calling I_cov and zero_mean from functions.R
L <- 1e3*I_cov(u)  # ditto
A0 <- zero_mean((r-u),u) # ditto
e <- zero_mean(r,p) # ditto
M <- 1e-3*I_cov(p) # ditto
K_inv <- cholsolve(K) ## this inverse works + calling cholsolve from functions.R
L_inv <- cholsolve(L) ## this inverse works

#####------------------------------------------------------------------------------------------------
# beta_ols
model <- lm((Y-mu_true)~X-1)
beta_ols <-t(model$coefficients) # first freq esimate
err_ols <- norm(beta_ols-beta_true, type="2")

# beta_envlp
#library(Renvlp)
envlp <- Renvlp::env(X, Y, u)
beta_env <- envlp$beta
err_env <- norm(beta_env-beta_true, type="2") ## SECOND FREQ ESTIMATE


G_env <- envlp$Gamma
A_env <- get_A(G_env)
# }}}FIRST CHUNK OF CODE 

##--------------------------------------------------------------------------------------------------
#### constants (indep of A) used in log_post_w, d_lpda_1,2,3, d_lpda.

#{{{ second part of the code
# no. of bootstraps, NB
NB <- 100
# generating w:
w <- list()
w <- sapply(1:NB, function(i) rexp(n))
#----------------------------------------------------------------------------------------------------
Nw <- lapply(X=c(1:NB), function(x) return(Nwf(w[,x]))) # calling Nwf 
G1 <- lapply(X=c(1:NB), function(x) return(G1f(w[,x]))) # calling G1f
Gwt <- lapply(X=c(1:NB), function(x) return(Gwtf(w[,x]))) # calling Gwtf
#----------------------------------------------------------------------------------------------------
### OPTIMIZATION IN THE FOR LOOP using foreach 

library(foreach, quietly = TRUE)
library(parallel, quietly = TRUE)
library(doParallel, quietly = TRUE)


# multiple starting points for non-convex optimization
MIP <- 5 ## MAX_INITIAL_POINTS


## optimization & initialization
A_opt <- list()
A_init <- list()

## function getting A_max:
A_max <- function(MIP,k){ ## this needs all the functions which make up F_minus, GR_minus, f
 print("I am in A_max function")#new today
 A_candidate <- list()
 f_val <- vector()
 
 for(iter in 1:MIP){
 print(c("starting point number", iter)) #new today 
 if(iter>1)  {
        A_init <- rnorm((r-u)*u,0,1e3)
      } else if (iter==1){
        A_init <- fastmatrix::vec(A_env)
        print(A_init) # new today
      }
      result = optim(par = A_init, 
                     fn = function(x) return(F_minus(x, i=k)),
                     gr = function(x) return(GR_minus(x, i=k)), 
                     method = "BFGS")
      A_candidate[[iter]] <- result$par
     print(c("A_candidate[[iter]]=", A_candidate[[iter]])) #new today
      f_val[iter] <- f(A_candidate[[iter]],i=k)
    }
    index <- which.max(f_val[1:MIP])
    arg_max <- A_candidate[[index]]
   # print("I am in A_max function")#new today
   # print(arg_max) #new today
  return(arg_max)
}
# }}} SECOND CHUNK OF CODE

####----------------------------------------------------------------------------------------


# Filter for functions
functions_in_global <- all_objects[sapply(all_objects, function(x) is.function(get(x, envir = .GlobalEnv)))]


# making clusters
cl <- parallel::makeCluster(nclust)
doParallel::registerDoParallel(cl)

time <- system.time(
  
    
   A_opt<-  foreach (k = 1:NB, 
                     .packages = c("expm","fastmatrix",
                                   "matrixNormal"),
                     .export = functions_in_global) %dopar% {
    print(k) # new today
    #print(A_max(MIP,k)) #new today
    return(A_max(MIP,k))    
  }
)
parallel::stopCluster(cl)
##-----------------------------------------------------------------------------------------
### OBJECTS TO BE SAVED
object1 <- list("beta_ols"=beta_ols,
                "beta_env" = beta_env,
                "err_ols" = err_ols,
                "err_env" = err_env)
object2 <-comparisons_wbb(100,A_opt, time)
# object3 <- mcmc/benvlp estimates of beta, time taken for estimation and errors!

object <- append(object1,object2)

## THE OUTPUT
 object 
saveRDS(object, file = paste0("output_", n, "_", iter_number,  ".rds")) 
