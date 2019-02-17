import numpy as np
from scipy import sparse as sps
from scipy.sparse.linalg import spsolve
#from numba import jit
    
#@jit
def implicitFILT8(Fi=None,alfaf=None):

    N=len(Fi)
    # Tridiagonal matrix
    row = np.arange(1,N-1)
    col = np.arange(0,N-2)
    data = np.zeros(N-2)
    data[:]=alfaf
    A1=sps.csr_matrix((data, (row, col)), shape=(N, N))
    
    row = np.arange(1,N-1)
    col = np.arange(1,N-1)
    data = np.zeros(N-2)
    data[:]=1.
    Ad=sps.csr_matrix((data, (row, col)), shape=(N, N))

    
    row = np.arange(1,N-1)
    col = np.arange(2,N)
    data = np.zeros(N-2)
    data[:]=alfaf
    A2=sps.csr_matrix((data, (row, col)), shape=(N, N))
    
    A = A1 + Ad + A2
    
    A[0,0:4] = np.array( [4*alfaf+1 ,-5*alfaf,4*alfaf ,-alfaf] )
    
    A[N-1,N-4:N] = np.array( [-alfaf, 4*alfaf, -5*alfaf, 4*alfaf+1] )
    
    # Cubic extrapolation of Fi values
    F0  =  4*Fi[0]    -6*Fi[1]   +4*Fi[2]-Fi[3]
    Fm1 = 10*Fi[0]   -20*Fi[1]   +15*Fi[2]-4*Fi[3]
    Fm2 = 20*Fi[0]   -45*Fi[1]   +36*Fi[2]-10*Fi[3]
    Fm3 = 35*Fi[0]   -84*Fi[1]   +70*Fi[2]-20*Fi[3]
    FN1 =  4*Fi[N-1]  -6*Fi[N-2] +4*Fi[N-3]-Fi[3]
    FN2 = 10*Fi[N-1] -20*Fi[N-2] +15*Fi[N-3]-4*Fi[N-4]
    FN3 = 20*Fi[N-1] -45*Fi[N-2] +36*Fi[N-3]-10*Fi[N-4]
    FN4 = 35*Fi[N-1] -84*Fi[N-2] +70*Fi[N-3]-20*Fi[N-4]

    # Coefficients
    a0 = (93+70*alfaf)/128.
    a1 = (7+18*alfaf)/16.
    a2 = (-7+14*alfaf)/32.
    a3 = (1-2*alfaf)/16.
    a4 = (-1+2*alfaf)/128.

    b=np.zeros(N)
    # RHS vector
    b[0]   = np.matmul( 1./2.*np.array([a0, a1, a2, a3, a4]) , np.array( [ 2*Fi[0], Fi[1]+F0   , Fi[2]+Fm1  ,Fi[3]+Fm2  , Fi[4]+Fm3] ) )
    b[1]   = np.matmul( 1./2.*np.array([a0, a1, a2, a3, a4]) , np.array( [ 2*Fi[1], Fi[2]+Fi[0], Fi[3]+F0   ,Fi[4]+Fm1  , Fi[5]+Fm2] ) )
    b[2]   = np.matmul( 1./2.*np.array([a0, a1, a2, a3, a4]) , np.array( [ 2*Fi[2], Fi[3]+Fi[1], Fi[4]+Fi[0],Fi[5]+F0   , Fi[6]+Fm1] ) )
    b[3]   = np.matmul( 1./2.*np.array([a0, a1, a2, a3, a4]) , np.array( [ 2*Fi[3], Fi[4]+Fi[2], Fi[5]+Fi[1],Fi[6]+Fi[0], Fi[7]+F0 ] ) )
    b[N-4] = np.matmul( 1./2.*np.array([a0, a1, a2, a3, a4]) , np.array( [ 2*Fi[N-4], Fi[N-3]+Fi[N-5], Fi[N-2]+Fi[N-6], Fi[N-1]+Fi[N-7], FN1+Fi[N-8] ] ) )
    b[N-3] = np.matmul( 1./2.*np.array([a0, a1, a2, a3, a4]) , np.array( [ 2*Fi[N-3], Fi[N-3]+Fi[N-4], Fi[N-1]+Fi[N-5], FN1    +Fi[N-6], FN2+Fi[N-7] ] ) )
    b[N-2] = np.matmul( 1./2.*np.array([a0, a1, a2, a3, a4]) , np.array( [ 2*Fi[N-3], Fi[N-1]+Fi[N-3], FN1    +Fi[N-4], FN2    +Fi[N-5], FN3+Fi[N-6] ] ) )
    b[N-1] = np.matmul( 1./2.*np.array([a0, a1, a2, a3, a4]) , np.array( [ 2*Fi[N-1], FN1    +Fi[N-2], FN2    +Fi[N-3], FN3    +Fi[N-4], FN4+Fi[N-5] ] ) )


    b[4:N-4]= 1./2.*( a0*(2*Fi[4:N-4])+a1*(Fi[5:N-3]+Fi[3:N-5])+a2*(Fi[6:N-2]+Fi[2:N-6])+a3*(Fi[7:N-1]+Fi[1:N-7])+a4*(Fi[8:N]+Fi[0:N-8] ) ) 
 
    # Filtered vector
    filtFi = (spsolve( A, b ) )

    return filtFi
