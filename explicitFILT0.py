import numpy as np
#from numba import njit
    
#@njit
def explicitFILT0(Fi=None):

    # Explicit Shapiro filtering for vector Fi
# Zero order formula (p=0)
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Arguments :
#  Fi : cell averaged vector
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
    N=len(Fi)
    filtFi=np.zeros(N)
    # Coefficients
    a0=2.
    a1=1.
    # Filtered vector
    filtFi[0]=Fi[0]
    
    filtFi[1:N-1]=1./4.*( a0*Fi[1:N-1] +a1*(Fi[2:N]+Fi[0:N-2] ))

    filtFi[N-1]=Fi[N-1]
    return filtFi
