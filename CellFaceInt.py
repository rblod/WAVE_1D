import numpy as np 
from scipy.sparse.linalg import spsolve
#from numba import jit

    
#@jit
def CellFaceInt(dx=None,Fi=None,A=None,fj1=None,*args):

    # ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# 4th order Compact cell face values interpolation from cell averaged values (Lacor et al., 2004)
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Arguments :
#  dx : spatial grid resolution (m)
#  Fi : cell averaged values
#  fj1 : cell face value at left boundary
#  A : compact-scheme matrix
#  varargin : optional argument = { fjN , 'q' }
#             fjN : right boundary condition in fj
#             If fjN is not fixed cell face value at the right boundary is linearly extrapolated 
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

#    Fi=Qj[1:Ntot - 1]
#    A=ADi
#    fj1=qi1
#    varargin=qiNm1   
    
    n= 4 +len(args)

    N=len(Fi) - 1

    A[0,0:2]=np.array([1,1./4.])
    A[N-1,N-2:N]=np.array([1./4.,1])
    bq=np.zeros([N])
    if n == 5:
        fjN=args[0]
        bq[N-1]=3./4.*(Fi[N-1] + Fi[N]) - 1./4.*fjN
    else:
        if n == 4:
            A[N-1,N-2:N]=np.array([0.,3./2.])
            bq[N-1]=3./4.*(Fi[N-1] + Fi[N])
    
    # RHS vector
    fjn=0.
    bq[0]=1./4.*(3*(Fi[0] + Fi[1]) - fj1)
    bq[1:N-1]=3./4.*(Fi[1:N-1] + Fi[2:N]) 
    # Interpolated cell face values
    if n == 5:
        fj=np.concatenate([np.array([fj1]) ,spsolve(A,bq) ,np.array([fjN]) ])
    else:
        if (n == 4):
            fji=spsolve(A,bq.T).T
            fjN=-1*fji[N - 2]+2*fji[N-1]
            fj=np.concatenate([fj1,fji,np.array([fjN])])
 #   print np.shape(spsolve(A,bq))
 #   print np.shape(np.array([fj1]))
 #   print np.shape(fj)

    return fj
