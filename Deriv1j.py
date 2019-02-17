import numpy as np
from scipy.sparse.linalg import spsolve
from numpy import dot as dot
#from numba import jit


#@jit
def Deriv1j(fj=None,dx=None,A=None):
    # 4th order Compact Scheme 1st Derivative Calculation for j nodes 
# Cubic extrapolation for "ghost" cells
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Arguments :
#  fj : cell face values
#  dx : spatial grid resolution (m)
#  A : compact-scheme matrix
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
    N=len(fj)
    A[0,0:2]=np.array([1,3])
    A[N-1,N - 2:N]=np.array([3,1])
    
    b=np.zeros([N])
    # RHS vector
    b[0]=1./(6.*dx)*(-17*fj[0] + 9*fj[1] + 9*fj[2] - fj[3])
    b[1:N-1]= 3./(4.*dx)*(fj[2:N] - fj[0:N - 2])
    b[N-1]  =1./ (6.*dx)*(17*fj[N-1] - 9*fj[N-2] - 9*fj[N-3] + fj[N-4])
    # 1st derivative of vector fj
    D1fj=spsolve(A,b).T
    
    return D1fj
