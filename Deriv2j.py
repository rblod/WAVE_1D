import numpy as np
from scipy.sparse.linalg import spsolve
from numpy import dot as dot
#from numba import jit
    

#@jit
def Deriv2j(fj=None,dx=None,A=None):
    # 4th order Compact Scheme 2nd Derivative Calculation for j nodes
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Arguments :
#  fj : cell face values
#  dx : spatial grid resolution (m)
#  A : compact-scheme matrix
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
    N=len(fj)
    A[0,0:2]=np.array([1,11])    
    A[N-1,N-2:N]=np.array([11,1])
    
    b=np.zeros([N])
    b[0]=1./(dx ** 2)*(13*fj[0] - 27*fj[1] + 15*fj[2] - fj[3])
    b[1:N-1]=dot(6. / (dot(5,dx ** 2)),(fj[2:N] - dot(2,fj[1:N - 1]) + fj[0:N - 2]))
    b[N-1]=dot(1. / dx ** 2,(dot(13,fj[N-1]) - dot(27,fj[N-2]) + dot(15,fj[N-3]) - fj[N-4]))
    # 2nd derivative of vector Fj
    D2fj=spsolve(A,b).T

    return D2fj
