# EXPERIMENT: RUNNING A SUBSET OF THE NB number of bootstrap blocks in Py, all functions being tensor based

import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
from scipy.optimize import minimize

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Param:
    def __init__(self):
        self.NB = NB
        self.n = n
        self.p = p
        self.r = r
        self.u = u
        self.Psi = Psi
        self.Psi_0 = Psi0
        self.nu = nu
        self.nu_0 = nu0
        self.A0 = A0
        self.M = M
        self.K_inv = K_inv
        self.L_inv = L_inv
        self.A_test = A_test
        self.A_env = A_env
        self.init_pts = init_pts
        self.A_true = A_true

# Required params
par = Param()


# Convert inputs to tensors and move to device
#X = torch.tensor(X, dtype=torch.float64, device=device)
#Y = torch.tensor(Y, dtype=torch.float64, device=device)
Nw_subvec = torch.tensor(Nw_subvec, dtype=torch.float64, device=device)
g1_submat = torch.tensor(g1_submat, dtype = torch.float64, device = device)
gwt_submat = torch.tensor(gwt_submat, dtype = torch.float64, device = device)
par.Psi = torch.tensor(par.Psi, dtype=torch.float64, device=device)
par.Psi_0 = torch.tensor(par.Psi_0, dtype=torch.float64, device=device)
par.K_inv = torch.tensor(par.K_inv, dtype=torch.float64, device=device)
par.L_inv = torch.tensor(par.L_inv, dtype=torch.float64, device=device)
par.A0 = torch.tensor(par.A0, dtype=torch.float64, device=device)
par.M = torch.tensor(par.M, dtype=torch.float64, device=device)
par.A_env = torch.tensor(par.A_env, dtype=torch.float64, device=device)
par.init_pts = torch.tensor(par.init_pts, dtype=torch.float64, device=device)
par.A_true = torch.tensor(par.A_true, dtype=torch.float64, device=device)

# Print types of objects received from R
print("---------------------------------------------------")
#print("Checking types of objects received from R:")
#print(f"Type of NB: {type(NB)}")
#print(f"Type of n: {type(n)}")
#print(f"Type of p: {type(p)}")
#print(f"Type of r: {type(r)}")
#print(f"Type of u: {type(u)}")
#print(f"Type of Psi: {type(par.Psi)}")
#print(f"Type of Psi_0: {type(par.Psi-0)}")
#print(f"Type of nu: {type(nu)}")
#print(f"Type of nu_0: {type(nu0)}")
#print(f"Type of A0: {type(par.A0)}")
#print(f"Type of M: {type(par.M)}")
#print(f"Type of K_inv: {type(par.K_inv)}")
#print(f"Type of L_inv: {type(par.L_inv)}")
#print(f"Type of A_test: {type(par.A_test)}")
#print(f"Type of A_env: {type(par.A_env)}")
#print(f"Type of init_pts: {type(par.init_pts)}")
#print(f"Type of A_true: {type(par.A_true)}")
#print(f"Type of X: {type(X)}")
#print(f"Type of Y: {type(Y)}")
print(f" size of Nw_subvec: {Nw_subvec.shape}")
#print(f"Type of g1_submat: {type(g1_submat)}")
#print(f"sixe of f1_submat: {g1_submat.size}")
#print(g1_submat)
#print("---------------------------------------------------")
#print("reached here")
def generate_gammas_from_A_torch(A): ## A is 2d tensor
    A = A.double()  # Ensure A is double
    dims = A.shape
    u = dims[1]
    r = sum(dims)
    U, S, V = torch.linalg.svd(A, full_matrices=True)
    S_squared = S**2
    I_plus_S_squared_inv_sqrt = 1 / torch.sqrt(1 + S_squared)
    pad_CAtCA = torch.ones(u, device=A.device, dtype=torch.float64)
    pad_CAtCA[:S.shape[0]] = I_plus_S_squared_inv_sqrt
    pad_DAtDA = torch.ones(r-u, device=A.device, dtype=torch.float64)
    pad_DAtDA[:S.shape[0]] = I_plus_S_squared_inv_sqrt
    CAtCA_minus_half = V @ torch.diag(pad_CAtCA) @ V.t()
    DAtDA_minus_half = U @ torch.diag(pad_DAtDA) @ U.t()
    eye_u = torch.eye(A.shape[1], device=A.device, dtype=torch.float64)
    eye_r_minus_u = torch.eye(A.shape[0], device=A.device, dtype=torch.float64)
    gamma = torch.cat((eye_u, A), dim=0) @ CAtCA_minus_half
    gamma0 = torch.cat((-A.t(), eye_r_minus_u), dim=0) @ DAtDA_minus_half
    return gamma.contiguous(), gamma0.contiguous()


def generateFunctions_listed(Nw, g1, gwt, params): # X =2d tensor, Y =2d tensor, w = an element of a tensor of order= order of w_subset
    
    #NW = torch.sum(w)  # Use torch.sum
    #G1 = get_g1(w) ## do this more efficiently in R or python.
    #GWT = get_gwt(w)
    nu = params.nu
    nu_0 = params.nu_0
    c1 = - (nu + Nw - 1) / 2
    c2 = - (nu_0 + Nw - 1) / 2

    def f_listed(A):
        Gm, Gm0 = generate_gammas_from_A_torch(A)
        t1 = Gm.T @ gwt @ Gm + par.Psi
        t2 = Gm0.T @ g1 @ Gm0 + par.Psi_0
        tmp3 = A - par.A0
        t3 = par.K_inv @ tmp3 @ par.L_inv @ tmp3.T ## line 238 of predenv_...
        final = c1 * torch.logdet(t1) + c2 * torch.logdet(t2) - torch.trace(t3) / 2
        return -final

    return f_listed

def closure(A_nparray, Nw, g1, gwt, ncol): ## A_nparray is a 1d tensor?
    nrow = A_nparray.size // ncol
    #print(nrow)
    #print(f"Type of A_nparray: {type(A_nparray)}") # is np.array
    #A_tensor = A_nparray.clone().detach().requires_grad_(True)
    A_tensor = torch.tensor(A_nparray, dtype=torch.float64, requires_grad=True)
    A_tensor = A_tensor.reshape(nrow, ncol)
    #print(A_tensor.shape)
    log_post_w = generateFunctions_listed(Nw, g1, gwt, par)
    with torch.enable_grad():
        f = log_post_w(A_tensor) ## A_tensor should be a 2d tensor
        grad = autograd.grad(f, A_tensor)[0]
    grad_flattened_col = grad.T.flatten()
    return f.detach().cpu(), grad_flattened_col.detach().cpu()

def minimize_wi(Nw, g1, gwt):

    A_init = par.A_env.clone().detach()  # Important: Create a copy
    #A_init = A_init.T.flatten()  # Transpose then flatten
    
    # Ensure A_init is 2D before transposing
    if A_init.ndim == 1:
        A_init = A_init.unsqueeze(0)  # Add a dimension if it's 1D
        print(" it is 1D")

    A_init = A_init.T.flatten() 
   # A_init = A_init.permute(1, 0).flatten()  # Transpose then flatten
    print(f"Type of A_init: {type(A_init)}")
    # First minimization
    result = minimize(fun=lambda x: closure(x, Nw, g1, gwt, par.u),
                      x0=A_init.cpu(),  # Move initial point to CPU
                      jac=True,
                      method='BFGS',
                      options={'gtol': 1e-3})
    f_opt = result.fun
    A_opt = result.x

    # Second minimization (inner loop)
    for Aj in par.init_pts:
        A_init = Aj.clone().detach()  # Important: Create a copy
        A_init = A_init.T.flatten()  # Transpose then flatten
        #A_init = A_init.permute(1, 0).flatten()  # Transpose then flatten
        #print("check shape of A_init")
        print(A_init.shape)
        result = minimize(fun=lambda x: closure(x, Nw, g1, gwt, par.u),
                          x0=A_init.cpu(),  # Move initial point to CPU
                          jac=True,
                          method='BFGS',
                          options={'gtol': 1e-3})
        fmin_j = result.fun
        Amin_j = result.x

        if fmin_j < f_opt:
            A_opt = Amin_j
            f_opt = fmin_j

    # Reshape and return the optimized A
    A_opt = A_opt.reshape(par.u, par.r - par.u).T
    return A_opt

# Run minimize_wi for wi in w_subset
A_opt_results = []
for Nw, g1, gwt in zip(Nw_subvec, g1_submat, gwt_submat):
    A_opt_wi = minimize_wi(Nw, g1, gwt)
    A_opt_results.append(A_opt_wi)

