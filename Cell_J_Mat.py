# Generated with SMOP  0.41
import numpy as np
from scipy import sparse as sps
#from numba import jit

#@jit
def Cell_J_Mat(N=None):

    # Fj to fj Matrix transformation 
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Arguments :
#  N : number of j nodes
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Tridiagonal matrix (N x N)
    row = np.arange(1,N-3)
    col = np.arange(0,N-4)
    data = np.zeros(N-4)
    data[:]=1./24.
    A1=sps.csr_matrix((data, (row, col)), shape=(N-2, N-2))

    row = np.arange(1,N-3)
    col = np.arange(1,N-3)
    data = np.zeros(N-4)
    data[:]=22./24.
    Ad=sps.csr_matrix((data, (row, col)), shape=(N-2, N-2))

    row = np.arange(1,N-3)
    col = np.arange(2,N-2)
    data = np.zeros(N-4)
    data[:]=1./24.
    A2=sps.csr_matrix((data, (row, col)), shape=(N-2, N-2))

    A=A1 + Ad + A2    
    
    A[0,0:2]=1./24.*np.array([22, 1])
    A[N-3,N-4:N-2]=1./24.*np.array([1, 22]) 
    
    return A
