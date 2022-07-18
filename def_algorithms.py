## Sparse deconvolution problem: y = Dx + n

##########################################################
### NuSPAN: Nonuniform Sparse Proximal Average Network ###
##########################################################

from torch.nn import functional as F

def soft_thresh(x, lambda1):
    out = torch.sign(x) * (F.relu(torch.abs(x) - lambda1.float()))
    return out

import torch
import torch.nn as nn
import numpy as np

def firm_thresh(x, lambda2, gama):
    gama = gama.float()
    lambda2 = lambda2.float()
    out = torch.sign(x) * torch.min(torch.abs(x), torch.max( (gama/(gama-1)) * (torch.abs(x) - lambda2), torch.zeros_like(x) ) )
    return out

def scad(x, lambda3, a):
    out = torch.where( (torch.abs(x) > a*lambda3), x, x )
    out = torch.where( (torch.abs(x) > 2 * lambda3) & (torch.abs(x) <= a*lambda3), (((a-1)*x - torch.sign(x)*a*lambda3)/(a-2)), out )
    out = torch.where( (torch.abs(x) <= 2 * lambda3), torch.sign(x) * torch.max( torch.abs(x) - lambda3, torch.zeros_like(x)), out )
    return out

# defining custom dataset
from torch.utils.data import Dataset
class dataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X = X.to(self.device)
        self.Y = Y.to(self.device)

    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx, :] # since the input to the datset is tensor

def support_metric(x, x_hat):                                                   # x: true reflectivity, x_hat: predicted reflectivity
    num = np.intersect1d(x.nonzero()[0], x_hat.nonzero()[0]).shape[0]           # number of unique non-zero values that are in both arrays
    den = max( x.nonzero()[0].shape[0], x_hat.nonzero()[0].shape[0] )           # total number of non-zero values (max of both arrays)
    return num/den                                                              # support metric = ~accuracy of prediction of number of spikes? (not amplitude or location). 1 best.

def pes(x, x_hat):                                                              # x: true reflectivity, x_hat: predicted reflectivity
    intersection = np.intersect1d(x.nonzero()[0], x_hat.nonzero()[0]).shape[0]  # number of unique non-zero values that are in both arrays
    den = max(x.nonzero()[0].shape[0], x_hat.nonzero()[0].shape[0])             # total number of non-zero values (max of both arrays)
    num = den - intersection
    return np.abs(num/den)                                                      # support metric = ~accuracy of prediction of number of spikes? (not amplitude, somewhat location). 1 best.

import shutil
def save_ckp(folder, state, is_best):
    f_path = '{}/checkpoint.pt'.format(folder)
    torch.save(state, f_path)
    if is_best:
        best_fpath = '{}/best_model.pt'.format(folder)
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

from torch.autograd import Variable
import torchvision
import scipy.linalg

### NuSPAN-1 (Type 1: uniform omega across components)
class nuspan1(nn.Module):
    def __init__(self, m, n, D, lambda1_int, mu1_int, gama1_int, nu1_int, a1_int, L, maxit, omega1=(1/3), omega2=(1/3), omega3=(1/3), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(nuspan1, self).__init__()

        self._We = nn.Linear(m, n, bias=False)
        self._S = nn.Linear(n, n, bias=False)
        self.lambda1 = nn.Parameter(lambda1_int*torch.ones(m, maxit))
        self.lambda2 = nn.Parameter(mu1_int*torch.ones(m, maxit))
        self.gama = nn.Parameter(gama1_int*torch.ones(m, maxit))
        self.lambda3 = nn.Parameter(nu1_int*torch.ones(m, maxit))
        self.a = nn.Parameter(a1_int*torch.ones(m, maxit))
        self.maxit = maxit
        self.D = D
        self.L = L
        self.device = device
        self.omega1 = omega1
        self.omega2 = omega2
        self.omega3 = omega3

    # custom weights initialization called on network
    def weights_init(self):
        
        D = self.D
        L = self.L
        
        We = torch.from_numpy((1 / L) * D.T)
        We = We.float().to(self.device)
        
        S = torch.from_numpy(np.eye(D.shape[1]) - (1 / L) * np.matmul(D.T, D))
        S = S.float().to(self.device)
        
        self._We.weight = nn.Parameter(We)
        self._S.weight = nn.Parameter(S)

    def forward(self, x):
        x = x.to(self.device)
        Z = self.omega1 * soft_thresh(self._We(x), self.lambda1[:, 0]) + self.omega2 * firm_thresh(self._We(x), self.lambda2[:, 0], self.gama[:, 0]) + self.omega3 * scad(self._We(x), self.lambda3[:, 0], self.a[:, 0])

        if self.maxit == 1:                
            return Z

        for i in range(self.maxit-1):
            Z = self.omega1 * soft_thresh(self._We(x) + self._S(Z), self.lambda1[:, i+1]) + self.omega2 * firm_thresh(self._We(x) + self._S(Z), self.lambda2[:, i+1], self.gama[:, i+1]) + self.omega3 * scad(self._We(x) + self._S(Z), self.lambda3[:, i+1], self.a[:, i+1])
        return Z

### NuSPAN-2 (Type 2: nonuniform omegas across components)
class nuspan2(nn.Module):
    def __init__(self, m, n, D, lambda1_int, mu1_int, gama1_int, nu1_int, a1_int, L, maxit, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(nuspan2, self).__init__()

        self._We = nn.Linear(m, n, bias=False)
        self._S = nn.Linear(n, n, bias=False)
        self.lambda1 = nn.Parameter(lambda1_int*torch.ones(m, maxit))
        self.lambda2 = nn.Parameter(mu1_int*torch.ones(m, maxit))
        self.gama = nn.Parameter(gama1_int*torch.ones(m, maxit))
        self.lambda3 = nn.Parameter(nu1_int*torch.ones(m, maxit))
        self.a = nn.Parameter(a1_int*torch.ones(m, maxit))
        self.maxit = maxit
        self.D = D
        self.L = L
        self.device = device
        self.alpha = nn.Parameter((1/3)*torch.ones(maxit, n, 3))

    # custom weights initialization called on network
    def weights_init(self):
        
        D = self.D
        L = self.L
        
        We = torch.from_numpy((1 / L) * D.T)
        We = We.float().to(self.device)
        
        S = torch.from_numpy(np.eye(D.shape[1]) - (1 / L) * np.matmul(D.T, D))
        S = S.float().to(self.device)
        
        self._We.weight = nn.Parameter(We)
        self._S.weight = nn.Parameter(S)

    def forward(self, x):
        x = x.to(self.device)
        softmaxed_omega = F.softmax(self.alpha, dim=2)

        Z = softmaxed_omega[0,:,0] * soft_thresh(self._We(x), self.lambda1[:, 0]) + softmaxed_omega[0,:,1] * firm_thresh(self._We(x), self.lambda2[:, 0], self.gama[:, 0]) + softmaxed_omega[0,:,2] * scad(self._We(x), self.lambda3[:, 0], self.a[:, 0])

        if self.maxit == 1:                
            return Z

        for i in range(self.maxit-1):
            Z = softmaxed_omega[i+1,:,0] * soft_thresh(self._We(x) + self._S(Z), self.lambda1[:, i+1]) + softmaxed_omega[i+1,:,1] * firm_thresh(self._We(x) + self._S(Z), self.lambda2[:, i+1], self.gama[:, i+1]) + softmaxed_omega[i+1,:,2] * scad(self._We(x) + self._S(Z), self.lambda3[:, i+1], self.a[:, i+1])
        return Z, softmaxed_omega


####################################
### BPI: Basis-Pursuit Inversion ###
####################################
'''
% Reference: Chen, S. S., Donoho, D. L., and Saunders, M. A. Atomic decomposition by basis pursuit.SIAM Review, 43(1):129–159,2001. https://doi.org/10.1137/S003614450037906X
% Based on MATLAB code SolveBP (https://sparselab.stanford.edu/): SparseLab2.1-Core > Solvers > SolveBP.m
% SolveBP: Solves a Basis Pursuit problem
% Usage
%	sol = SolveBP(A, y, N, maxIters, lambda, OptTol)
% Input
%	A           Either an explicit nxN matrix, with rank(A) = min(N,n) 
%               by assumption, or a string containing the name of a 
%               function implementing an implicit matrix (see below for 
%               details on the format of the function).
%	y           vector of length n.
%   N           length of solution vector. 
%	maxIters    maximum number of PDCO iterations to perform, default 20.
%   lambda      If 0 or omitted, Basis Pursuit is applied to the data, 
%               otherwise, Basis Pursuit Denoising is applied with 
%               parameter lambda (default 0). 
%	OptTol      Error tolerance, default 1e-3
% Outputs
%	 sol             solution of BP
% Description
%   SolveBP solves the basis pursuit problem
%      min ||x||_1 s.t. A*x = b
%   by reducing it to a linear program, and calling PDCO, a primal-dual 
%   log-barrier algorithm. Alternatively, if lambda ~= 0, it solves the
%   Basis Pursuit Denoising (BPDN) problem 
%      min lambda*||x||_1 + 1/2||b - A*x||_2^2
%   by transforming it to an SOCP, and calling PDCO.  
%   The matrix A can be either an explicit matrix, or an implicit operator
%   implemented as a function. If using the implicit form, the user should
%   provide the name of a function of the following format:
%     y = OperatorName(mode, m, n, x, I, dim)
%   This function gets as input a vector x and an index set I, and returns
%   y = A(:,I)*x if mode = 1, or y = A(:,I)'*x if mode = 2. 
%   A is the m by dim implicit matrix implemented by the function. I is a
%   subset of the columns of A, i.e. a subset of 1:dim of length n. x is a
%   vector of length n is mode = 1, or a vector of length m is mode = 2.
'''

def bpi(D_trace, y_noisy, N, maxit, lambd, tol):
    import os
    from os import path
    if not path.exists('mytemp'):
        os.mkdir('mytemp')

    import oct2py
    oc = oct2py.Oct2Py(temp_dir='mytemp')

    import numpy as np

    A = D_trace.copy()
    n = y_noisy.shape[0]
    n_pdco = 2 * N
    m_pdco = n

    # upper and lower bounds
    bl = np.zeros((n_pdco, 1))
    bu = np.inf * np.ones_like(bl)

    # generate the vector c
    c = lambd * np.ones_like(bl)

    # generate an initial guess
    x0 = np.ones_like(bl) / n_pdco
    y0 = np.zeros((m_pdco, 1))
    z0 = np.ones_like(bl) / n_pdco

    d1 = 1.e-4  # regularization parameters
    d2 = 1

    xsize = 1   # Estimate of norm(x,inf) at solution
    zsize = 1   # Estimate of norm(z,inf) at solution

    Phi = np.concatenate([A, -A], axis=1)

    ##### begin pdco
    ##### [xx, yy, zz, inform, PDitns, CGitns, time] = pdco(c,      Phi,    y,  bl, bu, d1, d2, options, x0, y0, z0, xsize, zsize)
    ##### function [x,y,z,inform,PDitns,CGitns,time] = pdco(Fname,  Aname,  b,  bl, bu, d1, d2, options, x0, y0, z0, xsize, zsize )
    ##### solution x_hat = xx[1:N] - xx[(N+1):(2*N)]
    Fname = c.copy()
    Aname = Phi.copy()
    b = y_noisy.copy()
    b = np.expand_dims(b, axis=1)
    m = b.shape[0]
    n = bl.shape[0]

    normb = np.linalg.norm(b, np.inf)
    normx0 = np.linalg.norm(x0, np.inf)
    normy0 = np.linalg.norm(y0, np.inf)
    normz0 = np.linalg.norm(z0, np.inf)

    ## Initialize
    true = 1
    false = 0
    zn = np.zeros((n, 1))
    nb = n + m
    nkkt = nb
    CGitns = 0
    inform = 0

    maxitn = maxit
    featol = tol
    opttol = tol
    steptol = 0.99
    stepSame = 0
    x0min = 0.1
    z0min = 1.0
    mu0 = 0.01
    Method = 1
    itnlim = 20 * min(m, n)
    atol1 = 1.e-3
    atol2 = 1e-15
    conlim = 1.e+12
    wait = 0

    ## Set other parameters
    kminor    = 0          # 1 stops after each iteration
    eta       = 1.e-4      # Linesearch tolerance for "sufficient descent"
    maxf      = 10         # Linesearch backtrack limit (function evaluations)
    maxfail   = 1          # Linesearch failure limit (consecutive iterations)
    bigcenter = 1.e+3      # mu is reduced if center < bigcenter
    thresh    = 1.e-8      # For sparse LU with Method=41

    ## Parameters for LSQR.
    atolmin   = np.spacing(1)   # Smallest atol if linesearch back-tracks
    btol      = 0               # Should be small (zero is ok)
    show      = false           # Controls LSQR iteration log
    gamma     = np.max(d1)
    delta     = np.max(d2)

    low, upp, fix = pdxxxbounds(bl, bu)

    low = low[0]
    nfix = len(fix[0])
    if nfix > 0:
        x1 = zn
        x1[fix] = bl[fix]
        r1 = Aname @ x1
        b = b - r1

    # # % Scale the input data.
    # # % The scaled variables are
    # # %    xbar     = x/beta,
    # # %    ybar     = y/zeta,
    # # %    zbar     = z/zeta.
    # # % Define
    # # %    theta    = beta*zeta;
    # # % The scaled function is
    # # %    phibar   = ( 1   /theta) fbar(beta*xbar),
    # # %    gradient = (beta /theta) grad,
    # # %    Hessian  = (beta2/theta) hess.

    beta = xsize        # beta scales b, x
    if beta==0:
        beta = 1

    zeta = zsize        # zeta scales y, z
    if zeta==0:
        zeta=1

    theta = beta * zeta   # theta scales obj.(theta could be anything, but theta = beta*zeta makes scaled grad = grad/zeta = 1 approximately if zeta is chosen right.)

    bl[fix] = bl[fix] / beta
    bu[fix] = bu[fix] / beta
    bl[low] = bl[low] / beta
    bu[upp] = bu[upp] / beta
    d1      = d1 * ( beta / np.sqrt(theta) )
    d2      = d2 * ( np.sqrt(theta) / beta )

    beta2   = beta**2
    b       = b / beta
    y0      = y0/zeta
    x0      = x0 / beta
    z0      = z0 / zeta;

    ## Initialize vectors that are not fully used if bounds are missing.
    rL  = zn.copy()
    rU  = zn.copy()
    cL  = zn.copy()
    cU  = zn.copy()
    x1  = zn.copy()
    x2  = zn.copy()
    z1  = zn.copy()
    z2  = zn.copy()
    dx1 = zn.copy()
    dx2 = zn.copy()
    dz1 = zn.copy()
    dz2 = zn.copy()

    ## Initialize x, y, z1, z2, objective, etc.
    x       = x0.copy()
    y       = y0.copy()
    x[fix]  = bl[fix]

    for i in low:
        x[i]  = max( x[i], bl[i] )

    x1_list = []
    z1_list = []
    for i in low:
        # x1_one = x[i,0] - bl[i,0]
        # x1_max = max( x1_one, x0min )
        
        x1_list.append( max( x[i,0] - bl[i,0], x0min ) )
        z1_list.append( max( z0[i,0], z0min ) )

    x1 = np.array(x1_list)
    z1 = np.array(z1_list)

    x1 = np.expand_dims( x1, axis=1 )
    z1 = np.expand_dims( z1, axis=1 )

    if len(upp[0]) != 0:
        for i in upp:
            x[i]  = min( x[i], bu[i] )
            x2[i] = max( bu[i] - x[i], x0min )
            z1[i] = max( -z0[i], z0min )

    obj = ( Fname.T @ x ) * beta
    obj = obj[0,0]
    grad = Fname
    hess = np.zeros((n, 1))
    obj = obj / theta                               # Scaled obj.
    grad = grad * ( beta / theta ) + ( d1**2 * x)   # grad includes x regularization.
    H = hess * ( beta2 / theta ) + (d1**2)          # H includes x regularization.

    # # Compute primal and dual residuals:
    # #     r1 =  b - A*x - d2.^2*y
    # #     r2 =  grad - A'*y + (z2-z1)
    # #     rL =  bl - x + x1
    # #     rU = -bu + x + x2

    r1, r2, rL, rU, Pinf, Dinf = pdxxxresid1(Aname,fix,low,upp,b,bl,bu,d1,d2,grad,rL,rU,x,x1,x2,y,z1,z2)

    # # Initialize mu and complementarity residuals:
    # #     cL   = mu*e - X1*z1.
    # #     cU   = mu*e - X2*z2.

    mufirst = mu0
    mulast  = 0.1 * opttol
    mufirst = max( mufirst, mulast )
    mu      = mufirst

    cL, cU, center, Cinf, Cinf0, x1z1, x2z2 = pdxxxresid2(mu,low,upp,cL,cU,x1,x2,z1,z2)

    fmerit = pdxxxmerit(low, upp, r1, r2, rL, rU, cL, cU)

    ## Initialize other things
    PDitns    = 0
    converged = 0
    atol      = atol1
    atol2     = max( atol2, atolmin )
    atolmin   = atol2
    pdDDD2    = d2  # Global vector for diagonal matrix D2

    ## Iteration log
    stepx   = 0
    stepz   = 0
    nf      = 0
    itncg   = 0
    nfail   = 0

    regterm = (np.linalg.norm( d1 * x ))**2 + (np.linalg.norm( d2 * y ))**2
    objreg = obj + 0.5 * regterm
    objtrue = objreg * theta

    while np.logical_not(converged):
        PDitns  += 1
        r3norm  = max( Pinf, Dinf, Cinf )
        atol    = min( atol, 0.1*r3norm )
        atol    = max( atol, atolmin )

        # -----------------------------------------------------------------
        #   Solve (*) for dy.
        # -----------------------------------------------------------------
        #   Define a damped Newton iteration for solving f = 0,
        #   keeping  x1, x2, z1, z2 > 0.  We eliminate dx1, dx2, dz1, dz2
        #   to obtain the system
        # 
        #   [-H2  A' ] [dx] = [w ],   H2 = H + D1^2 + X1inv Z1 + X2inv Z2,
        #   [ A  D2^2] [dy] = [r1]    w  = r2 - X1inv(cL + Z1 rL)
        #                                     + X2inv(cU + Z2 rU),
        # 
        #   which is equivalent to the least-squares problem
        # 
        #      min || [ D A']dy  -  [  D w   ] ||,   D = H2^{-1/2}.     (*)
        #          || [  D2 ]       [D2inv r1] ||
        # -----------------------------------------------------------------

        H[low]  = H[low] + np.divide(z1[low], x1[low])
        H[upp]  = H[upp] + np.divide(z2[upp], x2[upp])
        w       = r2.copy()
        w[low]  = w[low] - np.divide( (cL[low] + z1[low]*rL[low]), x1[low] )
        w[upp]  = w[upp] - np.divide( (cU[upp] + z2[upp]*rU[upp]), x2[upp] )

        H       = np.divide(1, H)     # H is now Hinv (Note!)
        H[fix]  = 0
        D       = np.sqrt(H.copy())
        pdDDD1 = D.copy()
        explicitA = 1
        rw = np.hstack( (explicitA, Method, m, n, 0, 0, 0) )   # Passed to LSQR

        ### if Method == 1, use chol to get dy
        import scipy
        diags = np.array([0])
        AD = Aname @ scipy.sparse.diags(D.T, diags, shape=(n, n)).toarray()
        ADDA = ( AD.copy() @ AD.copy().T ) + scipy.sparse.diags(d2**2 * np.ones((m)), 0, shape=(m, m)).toarray()

        if PDitns==1:
            P = oc.symamd(ADDA)
            oc.close
        
        Q = (P[0].copy()-1).astype(int)
        ADDAP = ADDA[np.r_[Q],:][:,np.r_[Q]] 
        R = oc.chol(ADDAP)

        rhs = Aname @ (H*w) + r1
        dy = np.linalg.solve(R, np.linalg.solve(R.T, rhs[Q]) )
        dy[Q] = dy.copy()

        # # dy is now known.  Get dx, dx1, dx2, dz1, dz2
        grad = Aname.T @ dy
        grad[fix] = 0
        dx = H * (grad - w)
        dx1[low] = - rL[low] + dx[low]
        dx2[upp] = - rU[upp] - dx[upp]
        dz1[low] =  ( cL[low] - z1[low]*dx1[low] ) / x1[low]
        dz2[upp] =  ( cU[upp] - z2[upp]*dx2[upp] ) / x2[upp]

        stepx1 = pdxxxstep( x1[low], dx1[low] )
        stepx2 = pdxxxstep( x2[upp], dx2[upp] )
        stepz1 = pdxxxstep( z1[low], dz1[low] )
        stepz2 = pdxxxstep( z2[upp], dz2[upp] )
        stepx  = min( stepx1, stepx2 )
        stepz  = min( stepz1, stepz2 )
        stepx  = min( steptol*stepx, 1 )
        stepz  = min( steptol*stepz, 1 )

        if stepSame:                    # For NLPs, force same step
            stepx = min( stepx, stepz )   # (true Newton method)
            stepz = stepx

        ## Backtracking linesearch
        fail = True
        nf = 0

        while nf < maxf:
            nf      += 1
            x       = x.copy() + stepx * dx.copy()
            y       = y.copy() + stepz * dy.copy()
            x1[low] = x1[low] + stepx*dx1[low]
            x2[upp] = x2[upp] + stepx*dx2[upp]
            z1[low] = z1[low] + stepz*dz1[low]
            z2[upp] = z2[upp] + stepz*dz2[upp]

            obj = (Fname.T @ x) * beta
            obj = obj[0,0]
            grad = Fname
            hess = np.zeros((n,1))

            obj = obj/theta
            grad = grad*(beta/theta) + (d1**2)*x
            H = hess*(beta2/theta) + (d1**2)

            r1, r2, rL, rU, Pinf, Dinf = pdxxxresid1(Aname,fix,low,upp,b,bl,bu,d1,d2,grad,rL,rU,x,x1,x2,y,z1,z2)
            
            cL, cU, center, Cinf, Cinf0, x1z1, x2z2 = pdxxxresid2(mu,low,upp,cL,cU,x1,x2,z1,z2)
            
            fmeritnew = pdxxxmerit(low, upp, r1, r2, rL, rU, cL, cU)

            step = min(stepx, stepz)

            if fmeritnew <= (1 - eta*step)*fmerit:
                fail = False
                break

            # # Merit function didn't decrease. Restore variables to previous values. (This introduces a little error, but save lots of space.)

            x       = x.copy() - stepx * dx.copy()
            y       = y.copy() - stepz * dy.copy()
            x1[low] = x1.copy()[low] - stepx*dx1[low]
            x2[upp] = x2[upp] - stepx*dx2[upp]
            z1[low] = z1[low] - stepz*dz1[low]
            z2[upp] = z2[upp] - stepz*dz2[upp]
            
            # # Back-track. If it's the first time, make stepx and stepz the same.

            if nf==1 and stepx != stepz:
                stepx = step
            elif nf < maxf:
                stepx = stepx / 2
            
            stepz = stepx

        if fail:
            nfail = nfail + 1
        else:
            nfail = 0

        ## Set convergence measures

        regterm = np.linalg.norm( d1 * x )**2 + np.linalg.norm( d2 * y )**2
        objreg = obj + 0.5*regterm
        objtrue = objreg * theta

        primalfeas    = Pinf   <= featol
        dualfeas      = Dinf   <= featol
        complementary = Cinf0  <= opttol
        enough        = PDitns >= 4                # Prevent premature termination
        converged     = primalfeas & dualfeas & complementary & enough

        ## Test for termination
        if converged:
            # print('Converged')
            inform = 4
        elif PDitns >= maxitn:
            # print('Too many iterations')
            inform = 1
            break
        elif nfail >= maxfail:
            print('Too many linesearch failures')
            inform = 2
            break
        elif step <= 1.e-10:
            print('Step lengths too small')
            break
        else:
            ## Reduce mu, and reset certain residuals

            stepmu = min( stepx, stepz )
            stepmu = min( stepmu, steptol )
            mu_old  = mu
            mu     = mu - stepmu * mu

            if center >= bigcenter:
                mu = mu_old

            mu = max( mu, mulast )
            cL, cU, center, Cinf, Cinf0, x1z1, x2z2 = pdxxxresid2(mu,low,upp,cL,cU,x1,x2,z1,z2)
            fmerit = pdxxxmerit(low, upp, r1, r2, rL, rU, cL, cU)

            atolold = atol

            if nf > 2 or step <= 0.1:
                atol = atolold * 0.1

    ## End of main loop

    x[fix] = 0
    z = np.zeros((n, 1))
    z[low] = z1[low]
    z[upp] = z[upp] - z2[upp]

    bl[fix] = bl[fix]*beta
    bu[fix] = bu[fix]*beta
    bl[low] = bl[low]*beta
    bu[upp] = bu[upp]*beta

    x[fix] = bl[fix]

    b = b * beta
    if nfix > 0:
        x1      = np.zeros((n, 1))
        x1[fix] = bl[fix]
        b = b + r1

    obj = Fname.T @ x
    grad = Fname
    hess = np.zeros((n, 1))

    z = grad - Aname.T @ y

    xx = x.copy()
    yy = y.copy()
    zz = z.copy()

    x_hat = xx[ 0:N ] - xx[ (N) : ((2*N)) ]

    return x_hat

def pdxxxbounds(bl, bu):
    # % Categorize various types of bounds.
    # % pos overlaps with low.
    # % neg overlaps with upp.
    # % two overlaps with low and upp.
    # % fix and free are disjoint from all other sets.
    import numpy as np
    bigL = -9.9e+19
    bigU = 9.9e+19
    pos = np.where( np.logical_and( bl==0, bu>=bigU ) )
    neg = np.where( np.logical_and( bl<=bigL, bu==0 ) )
    low = np.where( np.logical_and( bl>bigL, bl<bu ) )
    upp = np.where( np.logical_and( bu<bigU, bl<bu ) )
    two = np.where( np.logical_and( bl>bigL, bu<bigU, bl<bu ) )
    fix = np.where( bl==bu )
    free = np.where( np.logical_and(  bl<=bigL, bu>=bigU ) )

    return low, upp, fix

def pdxxxresid1(Aname,fix,low,upp,b,bl,bu,d1,d2,grad,rL,rU,x,x1,x2,y,z1,z2):
    m = len(b)
    n = len(bl)
    x[fix] = 0
    r1 = Aname @ x
    r2 = Aname.T @ y

    r1 = b - r1 - (d2**2)*y
    r2 = grad - r2
    r2[fix] = 0
    
    if len(upp[0]) != 0:
        r2[upp] = r2[upp] + z2[upp]
    
    r2[low] = r2[low] - z1[low]
    rL[low] = ( bl[low] - x[low] ) + x1[low]
    
    if len(upp[0]) != 0:
        rU[upp] = ( -bu[upp] + x[upp] ) + x2[upp]

    import numpy as np

    if len(upp[0]) == 0:
        a = 0
    else:
        a = np.linalg.norm( rU[upp], np.inf)

    Pinf = max( np.linalg.norm( r1, np.inf), np.linalg.norm( rL[low], np.inf), a )
    Dinf = np.linalg.norm( r2, np.inf )
    Pinf = max( Pinf, 1.e-99 )
    Dinf = max( Dinf, 1.e-99 )

    return r1, r2, rL, rU, Pinf, Dinf

def pdxxxresid2(mu, low, upp, cL, cU, x1, x2, z1, z2):
    x1z1 = x1[low] * z1[low]
    x2z2 = x2[upp] * z2[upp]
    cL[low] = mu - x1z1
    cU[upp] = mu - x2z2

    import numpy as np

    # import os
    # os.mkdir('mytemp')
    # import oct2py
    # oc = oct2py.Oct2Py(temp_dir='mytemp')

    if x2z2.size == 0:
        maxXz = np.max(x1z1)
        minXz = np.min(x1z1)
    else:
        maxXz = max( np.concatenate([np.max(x1z1), np.max(x2z2)], axis=1) )
        minXz = min( np.concatenate([np.min(x1z1), np.min(x2z2)], axis=1) )
    
    maxXz   = max( maxXz, 1e-99 )
    minXz   = max( minXz, 1e-99 )
    center  = maxXz / minXz

    if len(upp[0]) == 0:
        a = 0
        Cinf = np.max( np.linalg.norm(cL[low], np.inf) )
    else:
        a = np.linalg.norm(cU[upp],inf)
        Cinf    = max( np.concatenate([np.linalg.norm(cL[low], np.inf), a], axis=1) )

    Cinf0   = maxXz

    return cL, cU, center, Cinf, Cinf0, x1z1, x2z2

def pdxxxmerit(low, upp, r1, r2, rL, rU, cL, cU):
    import numpy as np

    if len(upp[0]) == 0:
        # f = np.concatenate( [ np.linalg.norm(r1), np.linalg.norm(r2), np.linalg.norm(rL[low]), np.linalg.norm(cL[low]) ], axis=1)
        f = np.hstack( ( np.linalg.norm(r1), np.linalg.norm(r2), np.linalg.norm(rL[low]), 0, np.linalg.norm(cL[low]), 0 ) )
    else:
        a = np.linalg.norm(rU[upp])
        b = np.linalg.norm(cU[upp])
        f = np.hstack( ( np.linalg.norm(r1), np.linalg.norm(r2), np.linalg.norm(rL[low]), a, np.linalg.norm(cL[low]), b ) )

    fmerit = np.linalg.norm(f)

    return fmerit

def pdxxxstep(x, dx):
    step = 1.e+20
    import numpy as np
    blocking = np.where( dx < 0 )
    if len(blocking) > 0:
        steps = x[blocking] / (-dx[blocking])
        if len(steps) != 0:
            step = np.min(steps)

    return step


##############################################################
### FISTA: Fast Iterative Shrinkage-Thresholding Algorithm ###
##############################################################
import numpy as np

# Soft-threshold
def soft_thresh_istas(x, threshold):        # (sparse reflectivity vector, threshold)
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.)

# FISTA
def fista(D, y, lambd, maxit):              # (dictionary/kernel matrix, observation/measurement, regularization parameter (lambd), maximum iterations)
    x_hat = np.zeros(D.shape[1])                # sparse reflectivity vector; D.shape = m*n; y.shape = m*1; x_hat.shape = n*1
    # L = scipy.linalg.norm(D) ** 2         # Lipschitz constant (omega) # default ord (None) implies Frobenius norm (for matrices) or 2-norm (for vectors)
    L = scipy.linalg.norm(D, ord=2) ** 2    # ord=2 implies 2-norm (largest sing. value) for matrices. This gives the smallest Lipschitz constant. Converges to solution faster, with lower number of iterations
    t = 1
    z = x_hat.copy()
    
    for _ in range(maxit):
        x_prev = x_hat.copy()
        x_hat = soft_thresh_istas(z + (1 / L) * (np.dot(D.T, y - D.dot(z))), lambd / L)
        t0 = t
        t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2
        z = x_hat + ((t0 - 1.) / t) * (x_hat - x_prev)

    return x_hat


#######################################################################
### SBL-EM: Expectation-Maximization-based sparse Bayesian learning ###
#######################################################################
'''
% Reference: Wipf, D. P. and Rao, B. D. Sparse Bayesian learning for basis selection.IEEE Transactions on Signal Processing, 52(8):2153–2164, 2004. https://doi.org/10.1109/TSP.2004.831016
% Based on MATLAB code sparse_learning (http://dsp.ucsd.edu/~dwipf/sparse_learning.m) written by David Wipf
*************************************************************************
% *** PURPOSE *** 
% Implements generalized versions of SBL and FOCUSS for learning sparse
% representations from possibly overcomplete dictionaries.
%
% *** USAGE ***
% [mu,dmu,k,gamma] = sparse_learning(Phi,T,lambda,iters,flag1,flag2,flag3);
%
% *** INPUTS ***
% Phi (D)       = N X M dictionary
% T (y)         = N X L data matrix
% lambda        = scalar trade-off parameter (balances sparsity and data fit)
% iters         = maximum number of iterations
%
% flag1         = 0: fast Mackay-based SBL update rules
% flag1         = 1: fast EM-based SBL update rule
% flag1         = 2: traditional (slow but sometimes better) EM-based SBL update rule
% flag1         = [3 p]: FOCUSS algorithm using the p-valued quasi-norm
%
% flag2         = 0: regular initialization (equivalent to min. norm solution)
% flag2         = gamma0: initialize with gamma = gamma0, (M X 1) vector
%
% flag3         = display flag; 1 = show output, 0 = supress output
%
% *** OUTPUTS ***
% mu (x_hat)    = M X L matrix of weight estimates
% dmu           = delta-mu at convergence
% k             = number of iterations used
% gamma         = M X 1 vector of hyperparameter values
%
'''

def sbl_em(D, y, lambd, maxit, flag1, flag3):

    import numpy as np
    import scipy

    ## Control Parameters
    MIN_GAMMA = 1e-16
    MIN_DMU = 1e-12
    MAX_ITERS = maxit
    DISPLAY_FLAG = flag3        # = 0 for no runtime screen printouts, 1 for yes

    ## Initializations
    N, M = D.shape
    # N, L = y.shape
    N = y.shape[0]
    L = 1

    gamma = np.ones((M, 1))     # (epsilon) variance of sparse code x, initialized as an all-one vector; most elements go to zero after iteration

    keep_list = np.ones((M, 1))
    keep_list = np.arange(0, M, 1)
    keep_list = np.expand_dims(keep_list, axis=1)

    m = keep_list.shape[0]
    x_hat = np.zeros((M, L))       # mu
    dmu = -1
    k = 0                       # iterations

    # Learning loop
    while True:

        ## prune things as hyperparameters (gamma, sigma^2) go to zero
        if np.min(gamma) < MIN_GAMMA:
            index = []
            index = np.where(gamma.flatten() > MIN_GAMMA)
            gamma = gamma[index]
            D = D[:, index[0]]
            keep_list = keep_list[index]
            m = gamma.shape[0]
            
            if m == 0:
                break

        ## Compute new weights
        G = np.tile(np.sqrt(gamma).T, [N, 1])
        DG = D * G
        np.nan_to_num(DG)
        U, diag_S, Vh = np.linalg.svd(DG, full_matrices=False)
        U = -1 * U
        V = -1 * Vh.T

        U_scaled = U[ :, 0: (min(N,m)) ] * np.tile( np.divide( diag_S,(diag_S**2 + lambd + 1e-16) ).T , [N, 1])
        Xi = G.T * (V @ U_scaled.T)    # Variance (sigma^2)

        mu_old = x_hat
        x_hat = Xi @ y

        ## Update hyperparameters
        gamma_old = gamma
        mu2_bar = np.abs(x_hat)**2
        mu2_bar = np.expand_dims(mu2_bar, axis=1)
        
        if flag1 == 1:
            ## Fast SBL-EM
            R_diag = np.real( np.sum( np.multiply(Xi.T, D) ).T )
            gamma = np.sqrt( np.multiply( gamma, np.real( np.divide( mu2_bar, np.multiply(L,R_diag) ) ) ) )
        elif flag1 == 2:
            ## Traditional SBL-EM
            DG_sqr = np.multiply( DG, G )   # phi.T * phi
            Sigma_w_diag = np.real( np.subtract(gamma, ( np.sum(( np.multiply(Xi.T, DG_sqr) ), axis=0, keepdims=True) ).T ) )
            gamma = np.add(mu2_bar/L, Sigma_w_diag)

        else:
            print("Specify Fast or Traditional SBL-EM")

        ## Check stopping conditions, etc.
        k = k + 1   # Update iterations
        
        if DISPLAY_FLAG == 1:
            print('iterations: {}, number of coefficients {}, gamma change {}'.format(k, m, np.max( np.abs(gamma - gamma_old) ) ) )
        if k >= MAX_ITERS:
            break

        if x_hat.shape == mu_old.shape:
            dmu = np.max( np.max( np.abs(mu_old - x_hat) ) )
            if dmu < MIN_DMU:
                break

    # ## Expand weights, hyperparameters
    temp = np.zeros((M, 1))
    if m > 0:
        temp[keep_list,0] = gamma
    gamma = temp
    x_hat = np.expand_dims(x_hat, axis=1)
    temp = np.zeros((M, L))
    if m > 0:
        temp[keep_list, 0] = x_hat
    x_hat = temp

    return x_hat
