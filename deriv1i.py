import numpy as np
from scipy.sparse.linalg import spsolve
from numpy import dot as dot
#from numba import jit

#@jit
def deriv1i(fi=None,fj1=None,fjN=None,dx=None,A=None):
    # 4th order Compact Scheme 1st Derivative Calculation for i nodes
# Cubic extrapolation for cells
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Arguments :
#  fi : cell centered point wise values
#  fj1 : cell face value at left boundary
#  fjN : cell face value at right boundary
#  dx : spatial grid resolution (m)
#  A : compact-scheme matrix
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
    N=len(fi)
    A[0,0:2]=np.array([1,1. / 4.])
    A[N-1,N-2:N]=np.array([1./4.,1])

    # RHS vector
    b=np.zeros([N])
    b[0]=-14./(15.*dx)*fj1+1./(120.*dx)*np.dot(np.array([15, 100, -3]),np.array([fi[0],fi[1],fi[2]]))
    b[1:N - 1]=3./(4*dx)*(fi[2:N]-fi[0:N-2])
    b[N-1]=14./(15.*dx)*fjN+1./(120.*dx)*np.dot(np.array([3. ,-100, -15]),np.array([fi[N-3],fi[N-2],fi[N-1] ]))
    
    # 1st derivative of vector fi
    D1fi=spsolve(A,b).T    

    return D1fi
