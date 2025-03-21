## EXPERIMENT: Efficient computstion of G1 and GWT in R 

args <- commandArgs(trailingOnly = TRUE)

## the seed
seed <- as.integer(args[1])

## dimesions n,p,r,u 
n <- as.integer(args[2])
p <- as.integer(args[3])
r <- as.integer(args[4])
u <- as.integer(args[5])

##{{ new functions start here
xdx_optimized = function(X,w){

    xw = X*w
    xwbar = colSums(xw)
    xcw = scale(X, center = xwbar, scale = FALSE)
    xcw.t_dw_xcw = crossprod((xcw*w), xcw)

    return(xcw.t_dw_xcw)
}

xdx_vanilla = function(X,w){
    xw = X*w
    xbarw = colSums(xw)
    one_n = replicate(n,1)
    one.txbarw = crossprod(t(one_n), t(xbarw))
    xcw = X - one.txbarw
    xcw.t_dw_xcw = t(xcw)%*%diag(w)%*%xcw
    return(xcw.t_dw_xcw)
}

g1_gwt_optimized <- function(X, Y, w){
    
    xw = X*w
    xwbar = colSums(xw)
    xcw = scale(X, center = xwbar, scale = FALSE)
    #xcw.t_dw_xcw = crossprod((xw*w), xw)
    xcw_w = xcw*w
    
    Mwt = crossprod(xcw_w, xcw) + M
    #xcw.t_dw_xcw = crossprod((xcw*w), xcw)
    #Mwt = xcw.t_dw_xcw + M
    
    
    yw = Y*w
    ywbar = colSums(yw)
    ycw = scale(Y, center = ywbar, scale = FALSE)
    ycw.t_dw_ycw =  crossprod((ycw*w), ycw)
    
    xcw.t_dw_ycw = crossprod(xcw_w, ycw) ## delete later
    ewt = t(solve(Mwt) %*% crossprod(xcw_w, ycw))
    
    G1 = ycw.t_dw_ycw
    GWT = ycw.t_dw_ycw - ewt %*% Mwt %*% t(ewt)
    
    #return(list("G1" = G1, "GWT" = GWT, "ewt" =ewt, "xdy" = xcw.t_dw_ycw))
    return(list("G1" = G1, "GWT" = GWT))

}

g1_gwt_vanilla <- function(X,Y,w){

       Dw <- diag(w)
       nw <- sum(w)
       
       cw <- function(z,w){
       zbw <- colSums(Dw%*%z)/nw
       tmp1 <- z-rep(1,n)%*%t(zbw)
       return(tmp1)
        }
   
      Ycw <- cw(Y,w)  
      Xcw <- cw(X,w)
      YcwtDw<- t(Ycw)%*%Dw
      YcwtDwYcw <- YcwtDw%*%Ycw
      XcwtDwXcw <- t(Xcw)%*%Dw%*%Xcw
      #XcwtDwYcw <- t(Xcw)%*%t(YcwtDw)
      XcwtDwYcw <- t(Xcw)%*%Dw%*%Ycw
      
      g1 <- YcwtDwYcw
      Mwt <- XcwtDwXcw+M
      ewt <- t(solve(Mwt)%*%(XcwtDwYcw))
      gwt <- YcwtDwYcw-ewt%*%Mwt%*%t(ewt)
 
      #return(list("G1" = g1, "GWT" = gwt, "ewt" = ewt, "xdy" = XcwtDwYcw))
      return(list("G1" = g1, "GWT" = gwt))
}

##}} new functions end here

##{{ the older functions start here
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

##}} the older functions end here

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
  Y[j,] <- MASS::mvrnorm(1,mu = mean_y[,j],Sigma = var_y) ## script Y
}

w = rexp(n) # this is weight for one bootstrap!
w = w/sum(w) # standardized weight
###---------------------------------------------------------------------------------------



# Packages needed:----------------------------------------------------------------------
# List of required packages
#packages <- c("expm", "fastmatrix", "matrixNormal", "reticulate", "future.apply", "data.table", "Renvlp", "MASS")
packages <- c("expm", "fastmatrix", "matrixNormal", "Renvlp", "MASS")
# Load packages
invisible(lapply(packages, library, character.only = TRUE))


##----------------------------------------------------------------------------------------
# TRUE REGRESSION MODEL PARAMETERS:

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

# DATA:

set.seed(seed) #Set the random seed for data
X <- matrix(rnorm(n*p),nrow=n,ncol=p) ## script X

mean_y <- mu_true+Gm_true%*%eta_true%*%t(X)
var_y <-Gm_true%*%Omg_true%*%t(Gm_true)+Gm0_true%*%Omg0_true%*%t(Gm0_true)

Y <- matrix(0,nrow=n,ncol=r)

for(j in 1:n){
  Y[j,] <- MASS::mvrnorm(1,mu = mean_y[,j],Sigma = var_y)
}

###---------------------------------------------------------------------------------------
# hyper-params (fixed, seed doesn't affect this)

#Psi <- 0.001*diag(1,u)
#Psi0 <- 0.001*diag(1,(r-u))
#nu <- u 
#nu0 <- r-u
#K <- 1e3*diag(1,(r-u)) # calling I_cov and zero_mean from functions.R
#L <- 1e3*diag(1,u)  # ditto
#A0 <- matrix(0, ncol = (r-u), nrow =u)
#e <- matrix(0, ncol = r, nrow = p)
M <- 1e-3*diag(1,p) # ditto
#K_inv <- 1e-3*diag(1,(r-u)) ## this inverse works + calling cholsolve from functions.R
#L_inv <- 1e-3*diag(1,u)  ## this inverse works

#####------------------------------------------------------------------------------------------------

##-------------------------------------------------------------------------------------


t2 <- system.time({  tmp_vanilla = g1_gwt_vanilla(X,Y,w) })[3]/60
t1 <- system.time({   tmp_opt = g1_gwt_optimized(X,Y,w) })[3]/60
##  CHECK1: checking if the matrices are identical
print("are the matrices produced identical by the two methods?") #YES!  
print(identical(tmp_opt$G1, tmp_vanilla$G1))
print(identical(tmp_opt$GWT, tmp_vanilla$GWT))
#print(identical(tmp_opt$ewt, tmp_vanilla$ewt))
#print(identical(tmp_opt$xdy, tmp_vanilla$xdy))

## CHECK2: compare run time 
print("t1 and t2 (multiplied by 100 = no. of bootstraps =least number of times to run)")
print(c(t1*100,t2*100))
#print(t2)

print("ratio:")
print(t2/t1)



