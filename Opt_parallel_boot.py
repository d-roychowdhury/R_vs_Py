# EXPERIMENT: parallelizing NB number of bootstrap blocksin Py

import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
import matplotlib.pyplot as plt  # Import matplotlib
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor  #for parallelization.

class Param:
    def __init__ (self):
        self.NB = NB
        self.n = n
        self.p = p
        self.r = r
        self.u = u
        self.Psi            = Psi
        self.Psi_0          = Psi0
        self.nu             = nu
        self.nu_0           = nu0
        self.A0             = A0
        self.M              = M
        self.K_inv          = K_inv
        self.L_inv          = L_inv
        self.A_test         = A_test
        self.A_env	        = A_env
        self.init_pts       = init_pts
        self.A_true         = A_true
        

## required params
par = Param()

def generate_gammas_from_A_torch(A):
    A = A.double()  # Ensure A is double

    # Extract the dimensions of A
    dims = A.shape
    u = dims[1]
    r = sum(dims)

    # Compute SVD of A
    U, S, V = torch.linalg.svd(A, full_matrices=True)

    # Compute (I + S^2)^{-1/2}
    S_squared = S**2
    I_plus_S_squared_inv_sqrt = 1 / torch.sqrt(1 + S_squared)

    # Prepare padded versions for CAtCA and DAtDA
    pad_CAtCA = torch.ones(u, device=A.device, dtype=torch.float64)
    pad_CAtCA[:S.shape[0]] = I_plus_S_squared_inv_sqrt

    pad_DAtDA = torch.ones(r-u, device=A.device, dtype=torch.float64)
    pad_DAtDA[:S.shape[0]] = I_plus_S_squared_inv_sqrt

    # Compute CAtCA^{-1/2} = V(I + S^2)^{-1/2}V^T
    CAtCA_minus_half = V @ torch.diag(pad_CAtCA) @ V.t()

    # Compute DAtDA^{-1/2} = U(I + S^2)^{-1/2}U^T
    DAtDA_minus_half = U @ torch.diag(pad_DAtDA) @ U.t()

    # Compute gamma and gamma0
    eye_u = torch.eye(A.shape[1], device=A.device, dtype=torch.float64)
    eye_r_minus_u = torch.eye(A.shape[0], device=A.device, dtype=torch.float64)

    gamma = torch.cat((eye_u, A), dim=0) @ CAtCA_minus_half
    gamma0 = torch.cat((-A.t(), eye_r_minus_u), dim=0) @ DAtDA_minus_half

    return gamma.contiguous(), gamma0.contiguous()

## weighted centering function---------------------------------------------------------


def weighted_centering (data, weights):
    (length, width) = data.shape
    means = np.average (data, weights = weights.flatten(), axis = 0)
    means = means.reshape(1,-1)
    return data -  np.ones((length, 1)) @ means


## defining log_post_w as a fun of w--------------------------------------------------

def generateFunctions_listed (X, Y, w, params):
    
    # {{{ utility functions that depend on X,Y, w, etc.
    def get_g1(wi):
        Dw = np.diag(wi)
        Ycw = weighted_centering(Y, wi)
        return Ycw.T @ Dw @ Ycw

    def get_gwt(wi):
        Dw = np.diag(wi)
        Xcw = weighted_centering(X, wi)
        Ycw = weighted_centering(Y, wi)
        Mwt = Xcw.T @ Dw @ Xcw + params.M
        #NOTE inverse calculation here.
        ewt = ( np.linalg.inv(Mwt) @ Xcw.T @ Dw @ Ycw ).T
        gwt = Ycw.T @ Dw @ Ycw - ewt @ Mwt @ ewt.T
        return gwt
    # }}}

    NW  = sum (w)
    G1  = get_g1(w)
    GWT = get_gwt(w)
    c1 = - (par.nu  + NW - 1)/2
    c2 = - (par.nu_0 + NW - 1)/2
    
    # Convert all necessary variables to PyTorch tensors
    GWT = torch.tensor(GWT, dtype=torch.float64)
    G1 = torch.tensor(G1, dtype=torch.float64)
    Psi = torch.tensor(params.Psi, dtype=torch.float64)
    Psi_0 = torch.tensor(params.Psi_0, dtype=torch.float64)
    A0 = torch.tensor(params.A0, dtype=torch.float64)
    K_inv = torch.tensor(params.K_inv, dtype=torch.float64)
    L_inv = torch.tensor(params.L_inv, dtype=torch.float64)
    A0 = torch.tensor(params.A0, dtype=torch.float64)
    #print(f"dim of A0: {A0.shape}")
    def f_listed (A):
    
        # Convert all NumPy operations to PyTorch operations
        Gm, Gm0 = generate_gammas_from_A_torch(A) 
    
        t1 = Gm.t() @ GWT @ Gm + Psi
        #print(f"t1: {t1.shape}")
        t2 = Gm0.t() @ G1 @ Gm0 + Psi_0
        #print(f"t2: {t2.shape}")
        tmp3 = A - A0
        #print(f"tmp3: {tmp3.shape}")
        t3 = K_inv @ tmp3 @ L_inv @ tmp3.t()
        #print(f"t3: {t3.shape}")
        final = c1 * torch.logdet(t1) + c2 * torch.logdet(t2) - torch.trace(t3.contiguous()) / 2
        return -final

    return  f_listed

## Pass fun = closure with jac =True to minimize() of scipy.optimize
def closure(A_nparray, w, ncol): ## reshapes AND tensorizes the intput A_nparray
    
    ## convert A_nparray into a ((r-u),u) tensor from a ((r-u)*u, 1) np array
    nrow = A_nparray.size // ncol
    A_nparray = A_nparray.reshape(nrow, ncol, order='F') 
    A_tensor = torch.tensor(A_nparray, dtype=torch.float64, requires_grad=True)
    #print(f"A_init as tensor in closure--------------------: {A_tensor}")
    
    ## define log_post_wi
    log_post_w = generateFunctions_listed(X,Y,w,par) ## what is params?
    
    ## log_post_wi's derivative using autograd
    with torch.enable_grad():
              f = log_post_w(A_tensor)
              grad = autograd.grad(f, A_tensor)[0]
    ## Perform column-wise flattening of the gradient
    grad_flattened_col = grad.t().flatten()  # Transpose first, then flatten row-wise (which mimics column-major order)
    return f.detach().numpy(), grad_flattened_col.detach().numpy()


## THIS IS THE FUNCTION TO BE RUN IN PARALLEL:
#from concurrent.futures import ProcessPoolExecutor - specially needed here.

def minimize_wi(wi):
    """
    Encapsulates the minimization process for a single wi.
    This is the function that will be executed in parallel.
    """
    A_init = par.A_env.copy()  # Important: Create a copy to avoid modifying the original A_env
    A_init = A_init.reshape(-1, order='F').flatten()

    # First minimization
    result = minimize(fun=lambda x: closure(x, wi, par.u),
                      x0=A_init,
                      jac=True,
                      method='BFGS',
                      options={'gtol': 1e-3})
    f_opt = result.fun
    A_opt = result.x


    # Second minimization (inner loop)
    for Aj in par.init_pts:
        A_init = Aj.copy()  # Important: Create a copy to avoid modifying the original Aj
        #A_init = A_init.reshape(-1, order='F').flatten() #No longer needed, it is a vector
        result = minimize(fun=lambda x: closure(x, wi, par.u),
                          x0=A_init,
                          jac=True,
                          method='BFGS',
                          options={'gtol': 1e-3})
        fmin_j = result.fun
        Amin_j = result.x

        if fmin_j < f_opt:
            A_opt = Amin_j
            f_opt = fmin_j

    # Reshape and return the optimized A
    A_opt = A_opt.reshape((par.r - par.u), par.u, order='F')
    return A_opt, f_opt

# Parallel Execution using ProcessPoolExecutor
if __name__ == "__main__":
    num_processes = 4  # Adjust based on your CPU core count
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(minimize_wi, W))  # Use list to force evaluation

    # Process results from parallel execution
    for i, (A_opt, f_opt) in enumerate(results):
        print(f"Optimization results for W[{i}]:")
        print(f"  f_opt: {f_opt}")
        print(f"  A_opt shape: {A_opt.shape}")
        print(f" A_opt: {A_opt}")
        print(f" A_true: {A_true}")
        print(f" the distance from A_true: {np.linalg.norm(A_opt - par.A_true, ord='fro')}")
        print(f"  the distance of A_env A_true: {np.linalg.norm(par.A_env -par.A_true, ord ='fro')}")

