import numpy as np
from scipy import sparse as sps
#from numba import jit
    
#@jit
def Comp_2nd_Mat(N=None):

    # Compact scheme matrix generation for 2nd derivative estimate at j nodes
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Arguments :
#  N : number of j nodes
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Tridiagonal matrix

    row = np.arange(1,N-1)
    col = np.arange(0,N-2)
    data = np.zeros(N-2)
    data[:]=1./10.
    A1=sps.csr_matrix((data, (row, col)), shape=(N, N))

    row = np.arange(1,N-1)
    col = np.arange(1,N-1)
    data = np.zeros(N-2)
    data[:]=1.
    Ad=sps.csr_matrix((data, (row, col)), shape=(N, N))

    row = np.arange(1,N-1)
    col = np.arange(2,N)
    data = np.zeros(N-2)
    data[:]=1./10.
    A2=sps.csr_matrix((data, (row, col)), shape=(N, N))

    A =A1 + Ad +A2
    
    return A
