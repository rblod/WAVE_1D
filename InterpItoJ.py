import numpy as np
from scipy import sparse as sps
from scipy.sparse.linalg import spsolve
#from numba import jit

    
#@jit
def InterpItoJ(fi=None,fj1=None,fjN=None):

    # Fourth order implicit interpolation from i nodal values to j nodal values
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Arguments :
# 
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
    N=len(fi) - 1
    # Tridiagonal matrix

#m2 = sps.csr_matrix(([3,0], ([2,1], [1,3])), dtype=np.float)
#m2.eliminate_zeros()
    row = np.arange(1,N-1)
    col = np.arange(0,N-2)
    data = np.zeros(N-2)
    data[:]=1
    A1=sps.csr_matrix((data, (row, col)), shape=(N, N))
    
    row = np.arange(1,N-1)
    col = np.arange(1,N-1)
    data = np.zeros(N-2)
    data[:]=6
    Ad=sps.csr_matrix((data, (row, col)), shape=(N, N))
   
    row = np.arange(1,N-1)
    col = np.arange(2,N)
    data = np.zeros(N-2)
    data[:]=1
    A2=sps.csr_matrix((data, (row, col)), shape=(N, N))
    
    A=A1 + Ad + A2
  

    A[0,0]=6
    A[0,1]=1
   
    
  #  A[N-1,N-2:N]=np.array([1,6])
    A[N-1,N-2]=1
    A[N-1,N-1]=6


    b=4*(fi[0:N] + fi[1:N+1])
    b[0]=b[0] - fj1
    b[N-1]=b[N-1] - fjN

    fj=spsolve(A,b).T
    fj=np.concatenate((fj1,fj,fjN),axis=None)
    return fj
