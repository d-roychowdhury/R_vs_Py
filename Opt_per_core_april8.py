# EXPERIMENT: RUNNING A SUBSET OF THE NB number of bootstrap blocks in Py, all functions being tensor based

import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim
from scipy.optimize import minimize
import os
import tempfile

tmpdir = tempfile.mkdtemp()
os.environ["TMPDIR"] = tmpdir

print(f"Process {os.getpid()} is using TMPDIR: {os.environ.get('TMPDIR')}")

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
#print(f"size of A_env: {par.A_env.shape}")
#print(f"size of init_pts: {par.init_pts.shape}")

# Convert inputs to tensors and move to device

Nw_subvec = torch.tensor(Nw_subvec, dtype=torch.float64, device=device) ## changed this
g1_submat = torch.tensor(g1_submat, dtype = torch.float64, device = device) ## changed this
gwt_submat = torch.tensor(gwt_submat, dtype = torch.float64, device = device) ## changed this
par.Psi = torch.tensor(par.Psi, dtype=torch.float64, device=device)## changed this
par.Psi_0 = torch.tensor(par.Psi_0, dtype=torch.float64, device=device)## changed this
par.K_inv = torch.tensor(par.K_inv, dtype=torch.float64, device=device)## changed this
par.L_inv = torch.tensor(par.L_inv, dtype=torch.float64, device=device)## changed this
par.A0 = torch.tensor(par.A0, dtype=torch.float64, device=device)## changed this
par.M = torch.tensor(par.M, dtype=torch.float64 , device=device)## changed this
#par.A_env = torch.tensor(par.A_env, dtype=torch.float64, device=device)## changed this, and removed it
#par.init_pts = torch.tensor(par.init_pts, dtype=torch.float64, device=device)## changed this, and removed it
par.A_true = torch.tensor(par.A_true, dtype=torch.float64, device=device)## changed this

# Print types of objects received from R
#print("---------------------------------------------------")
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
#print(f" size of Nw_subvec: {Nw_subvec.shape}")
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
    nu = params.nu ## from torch.tensor(params.nu)
    nu_0 = params.nu_0
    c1 = - (nu + Nw - 1) / 2
    c2 = - (nu_0 + Nw - 1) / 2
    
    #print(Nw)
    #print(f"nu:{nu}")
    def f_listed(A):
        Gm, Gm0 = generate_gammas_from_A_torch(A)
        t1 = Gm.t() @ gwt @ Gm + par.Psi
        t2 = Gm0.t() @ g1 @ Gm0 + par.Psi_0
        tmp3 = A - par.A0
        t3 = par.K_inv @ tmp3 @ par.L_inv @ tmp3.t() ## line 238 of predenv_...
        final = c1 * torch.logdet(t1) + c2 * torch.logdet(t2) - torch.trace(t3) / 2
        return -final

    return f_listed

def closure(A_nparray, Nw, g1, gwt, ncol): ## A_nparray is a 1d tensor?
    nrow = A_nparray.size // ncol
    #print(nrow)
    #print(f"Type of A_nparray line 128: {type(A_nparray)}") # is np.array
    A_tensor = torch.tensor(A_nparray, dtype=torch.float64, requires_grad=True)

    A_tensor = A_tensor.reshape(nrow, ncol)
    #print(A_tensor.shape)
    #log_post_w = generateFunctions_listed(Nw, g1, gwt, par)
    log_post_w = generateFunctions_listed(Nw, g1.to(torch.float64), gwt.to(torch.float64), par)

    with torch.enable_grad():
        f = log_post_w(A_tensor) ## A_tensor should be a 2d tensor
        #grad = autograd.grad(f, A_tensor)[0]
        grad = autograd.grad(f, A_tensor)[0].to(torch.float64)
    grad_flattened_col = grad.T.flatten()
    #print("the f till convergence:") ##---> added this
    #print(f) ## ---> added this
   # return f.detach().cpu(), grad_flattened_col.detach().cpu()
    return f.detach().cpu(), grad_flattened_col.detach().cpu() ### ------> latest eta chilo!



def minimize_wi(Nw, g1, gwt):
    
   # print("-------------------start minimize-----------------------")
  
   # A_init = par.A_env.clone().detach()  # Important: Create a copy
   # A_init = A_init.T.flatten()  # Transpose then flatten
    #A_init = par.A_env
    A_init = par.A_env.astype(np.float64)
    A_init = A_init.reshape(par.u *(par.r -par.u))

    
    
    
    #A_init.requires_grad_()
   # A_init = A_init.permute(1, 0).flatten()  # Transpose then flatten
    #print(f"Type of A_init: {type(A_init)}")
    #print("the value of A_env")## ----> removed this
    #print(A_init)## ----> removed this
    # First minimization
    result = minimize(fun=lambda x: closure(x, Nw, g1, gwt, par.u),
                      #x0=A_init.cpu(),  # Move initial point to CPU
                      x0 = A_init,
                      jac=True,
                      method='BFGS',
                      options={'gtol': 1e-20})
    f_opt = result.fun
    A_opt = result.x
    
    #f_opt_check , _ = closure(A_opt, Nw, g1, gwt, par.u)
    #print(f"f_opt from A_env: {f_opt}") ##---> added this
    #print(f"f_opt from A_env: {f_opt:.20f}")  # Print with high precision
    #print("the A_opt starting with A_env")##---> added this
    #print(A_opt) ##---> added this
    #count_inner = 1
    #print(count_inner)
    # Second minimization (inner loop)
    for Aj in par.init_pts:
        #A_init_j = Aj # Important: Create a copy
        A_init_j = Aj.astype(np.float64)
        #print(f"Type of A_init_j line 178: {type(A_init_j)}")
        #A_init_j = Aj
       # print(f" the {count_inner}-th initial point")## ----> removed this
       # print(A_init_j)## ----> removed this
        #A_init = A_init.T.flatten()  # Transpose then flatten
        #A_init = A_init.permute(1, 0).flatten()  # Transpose then flatten
        #print("check shape of A_init")
        #print(A_init.shape) #---> removed this
        result = minimize(fun=lambda x: closure(x, Nw, g1, gwt, par.u),
                         # x0=A_init_j.cpu(),  # Move initial point to CPU
                          x0 = A_init_j,
                          jac=True,
                          method='BFGS',
                          options={'gtol': 1e-5})
        fmin_j = result.fun
        Amin_j = result.x
        #count_inner = count_inner +1

        if fmin_j < f_opt:
            #count_other_init = count_other_init ##---> added this
            #print("other init points lead to better optimization")##---> added this
            #print(f"fmin_j starting with init_pts: {fmin_j}") ##---> added this
            #print("the A_opt starting with other init_pts") ##---> added this
            #print(Amin_j) ##---> added this
            A_opt = Amin_j
            f_opt = fmin_j
    
  
    # Reshape and return the optimized A
    #A_opt = A_opt.reshape(par.u, par.r - par.u).T
    A_opt = A_opt.reshape( (par.r - par.u), par.u) ## changing 213 to 214 causes the removal of garbage values! HOW?

    del result
   # print(f"f_opt after looping through other init pts: {f_opt}") ##---> added this
   # print("-------------------finish minimize-----------------------")##---> added this
    return A_opt

# Run minimize_wi for wi in w_subset
A_opt_results = []
count =0  ##---> removed this, added again.
for Nw, g1, gwt in zip(Nw_subvec, g1_submat, gwt_submat):

    A_opt_wi = minimize_wi(Nw, g1, gwt)
   # print("----------------------------------------------- end of one optimization")
    #print(par.A_env)
    A_opt_results.append(A_opt_wi)
    count =count+1##---> removed this, added again
    print(count)##---> removed this, added again

