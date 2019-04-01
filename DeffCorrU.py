import numpy as np
from numpy import multiply
from scipy import sparse as sps
from scipy.sparse.linalg import spsolve
#from numba import jit

# DeffCorrU.m

    
#@jit
#@profile
def DeffCorrU(uj1=None,Qi=None,hi=None,hj=None,fj=None,Dxhj=None,Dxfj=None,Dxxfj=None,dx=None,Aq=None,Bq=None,uj_old=None,bD=None,alfa=None,tol=None,MaxIter=None,WDtol=None):

    # Arguments :
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ---------------------- Depth averaged velocity estimate from q values ------------------------------
# ------------------------- Deferred correction approach (iterative) ---------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Arguments
#  uj1 : left boundary condition for depth averaged velocity (m/s)
#  Qi : cell averaged q value (m/s)
#  hi : cell-centered values (i) for water depth (m)
#  hj : cell face water depth values (m)
#  fj : cell face bottom bathymetry (m)
#  Dxhj : 1st x-derivative of water depth
#  Dxfj : 1st x-derivative of bottom bathymetry
#  Dxxfj : 2nd x-derivative of bottom bathymetry
#  dx : spatial grid resolution (m)
#  Aq : compact-scheme matrix
#  Bq : 
#  uj_old : cell face depth averaged velocity values at previous time step (m/s)
#  bD : parameter for dispersive terms (active == 1 ; neglect == 0)
#  alfa : linear dispersion correction parameter
#  tol : acceptable relative error for Picard iterations 
#  MaxIter : maximum number of Picard iterations
#  WDtol : 
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
    N=hj.size
    rj=np.zeros(N)
    aj=np.zeros(N)
    bj=np.zeros(N)
    qj=np.zeros(N)
    uj_n=np.zeros(N)
    qj_n=np.zeros(N)

    hmask=np.zeros(N)
    hmask[np.where(hj!=0)]=1
    hmask[np.where(hj==0)]=0

    hi0=1./16.*np.matmul(np.array([35,-35,21,- 5]),hj[0:4])
    
# Right boudary extrapolated values of hi (linear extrapolation)

    hiN=np.matmul(np.array([-1./2.,3./2.]),hj[N-2:N])

    if bD == 1:
        rj = multiply((Dxhj + Dxfj),Dxfj) + multiply(1./2.*hj,Dxxfj)

# aj and bj coefficients
        aj[0]     = hi[0]**3/(3*hj[0]) + alfa*fj[0]**2
        aj[1:N-1] = hi[1:N-1]**3/(3*hj[1:N-1]) * hmask[1:N-1] + alfa*fj[1:N-1]**2 
        aj[N-1]   = hiN**3/(3*hj[N-1])*  hmask[N-1] + alfa*fj[N-1]**2
        bj[0] = hi0**3/(3*hj[0]) + alfa*fj[0]**2
        bj[1:N] = hi[0:N-1]**3/(3*hj[1:N])*hmask[1:N] + alfa*fj[1:N]**2
    
# Tridiagonal sparse matrix multiplying uj vector (N-2 x N-2)
    
    row = np.arange(1,N-3)
    col = np.arange(0,N-4)
    data = np.zeros(N-4)
    data[:]=(1+rj[2:N-2])/24. -1./dx**2*bj[2:N-2]
    A1=sps.csr_matrix((data, (row, col)), shape=(N-2, N-2))

    row = np.arange(1,N-3)
    col = np.arange(1,N-3)
    data = np.zeros(N-4)
    data[:]=(1+rj[2:N-2])*22./24.+1./dx**2*(aj[2:N-2]+bj[2:N-2])
    Ad=sps.csr_matrix((data, (row, col)), shape=(N-2, N-2))
 
    row = np.arange(1,N-3)
    col = np.arange(2,N-2)
    data = np.zeros(N-4)
    data[:]=(1+rj[2:N-2])/24. -1/dx**2*aj[2:N-2]
    A2=sps.csr_matrix((data, (row, col)), shape=(N-2,N -2))


    Au = A1 + Ad + A2
 
    Au[0,0] = (1+rj[1])/24.* 22 - 1/dx**2*( -(aj[1]+bj[1]) )
    Au[0,1] = (1+rj[1])/24.* 1- 1/dx**2*(  aj[1] )

#     #Au[0,0:2] = (1+rj[1])/24.* ([22, 1]) - 1/dx**2*( [-(aj[1]+bj[1]), aj[1]] )
#     #    Au[N-3,N-4:N-2] = (1+rj[N-2])/24.*([1, 22]) - 1/dx**2 * ( [ bj[N-2], -(aj[N-2]+bj[N-2]) ] )
    Au[N-3,N-4] = (1+rj[N-2])/24.*1 - 1./dx**2 * (  bj[N-2] )
    Au[N-3,N-3] = (1+rj[N-2])/24.*22 - 1./dx**2 * (  -(aj[N-2]+bj[N-2])  )
 
#     # ----------------------------------------------------------------------------------------------------
    
#     # ----------------------------------------------------------------------------------------------------
# # Pentadiagonal sparse matrix (N x N+4)
    row = np.arange(0,N)
    col = np.arange(0,N)
    data = np.zeros(N)
    data[:]=-1/(24*dx**2)*bj[0:N]
    K1=sps.csr_matrix((data, (row, col)), shape=(N,N+4))
   
    row = np.arange(0,N)
    col = np.arange(1,N+1)
    data = np.zeros(N)
    data[:]=1./(24.*dx**2)*(aj[0:N]+3*bj[0:N])
    K2=sps.csr_matrix((data, (row, col)), shape=(N,N+4))

    row = np.arange(0,N)
    col = np.arange(2,N+2)
    data = np.zeros(N)
    data[:]= -3/(24*dx**2)*(aj[0:N]+bj[0:N])
    Kd=sps.csr_matrix((data, (row, col)), shape=(N,N+4))

    row = np.arange(0,N)
    col = np.arange(3,N+3)
    data = np.zeros(N)
    data[:]= 1./(24.*dx**2)*(3*aj[0:N]+bj[0:N])
    K3=sps.csr_matrix((data, (row, col)), shape=(N,N+4))

    row = np.arange(0,N)
    col = np.arange(4,N+4)
    data = np.zeros(N)
    data[:]= -1/(24*dx**2)*aj[0:N]
    K4=sps.csr_matrix((data, (row, col)), shape=(N,N+4))
       
    KF=K1 + K2 + Kd + K3 + K4
    
# # ----------------------------------------------------------------------------------------------------
# # Right hand side vectors including boundary nodes
    
    row = np.arange(0,1)
    col = np.arange(0,1)
    data = np.zeros(1)
    data[:]= 1.
    b1=sps.csr_matrix((data, (row, col)), shape=(N-2,1))

    row = np.arange(N-3,N-2)
    col =  np.arange(0,1)
    data = np.zeros(1)
    data[:]= 1.
    b2=sps.csr_matrix((data, (row, col)), shape=(N-2,1))

    row = np.arange(0,1)
    col = np.arange(0,1)
    data = np.zeros(1 )
#  #   data = 1/26*([5, -4, 1])
    data[0] = 1./26.*5
    p1=sps.csr_matrix((data, (row, col)), shape=(1,N-2))
    p1[0,1]= 1./26.* -4
    p1[0,2]= 1./26.

    row = np.arange(0,1)
    col = np.arange(N-4,N-3)
    data = np.zeros(1 )
    data[0] = -1.
    p2=sps.csr_matrix((data, (row, col)), shape=(1,N-2))
    p2[0,N-3]=2. 

    bq = 3./4.*( Qi[0:N-2] + Qi[1:N-1])
    m1 = (1+rj[1])/24.   -1./dx**2*bj[1]
    m2 = (1+rj[N-2])/24. -1./dx**2*aj[N-2]

#     # ----------------------------------------------------------------------------------------------------
    Aq[0,0:2] = np.array([1., 1/4.])
    Aq[N-3,N-4:N-2]=np.array([1/4., 1.])
    AQ=Aq+1/4.*b1*p1+1/4.*b2*p2
    AU=Au+m2*b2*p2;
    BQ=Bq+1./24.*b1*p1+1/24.*b2*p2;
    P1=spsolve(AQ,bq)
    P2=spsolve(AQ,1./4.*b1)



# First uj vector
    uj=np.concatenate([ uj1,uj_old[1:N+1] ])


#  Left boundary extrapolated values of uj (cubic extrapolation)
    ujm1=np.matmul([10 ,-20 ,15, -4],uj[0:4])
    uj00=np.matmul([4 ,-6 ,4, -1],uj[0:4])

# Rigth boundary extrapolated values of uj (linear extrapolation)
    ujN1=np.matmul( [-1 ,2],uj[N-2:N] )
    ujN2=np.matmul( [-2 ,3],uj[N-2:N] )

 
#  Explicit derivative estimate
    Fj=KF * np.concatenate([ [ujm1],[uj00],uj,[ujN1],[ujN2]]) 

  # First qj vector
    qja=np.matmul (24./26. *((1+rj[0])/24.*np.array([1., 22., 1.])-1./dx**2*np.array([bj[0] ,-(aj[0]+bj[0]), aj[0]]) ),([ uj00 ,uj[0], uj[1] ]) ) -24./26.*Fj[0]
    qj[1:N-1]=(P1-P2*qja)
    qj[0] = 1./26.*np.matmul( np.array([5, -4, 1]),qj[1:4] ) + qja
    qj[N-1]=np.matmul(([-1, 2]),qj[N-3:N-1] )    # Linear extrapolation
      
 
# # Picard iterations
    i=0
    err=1
    beta1=1.0
    beta2=1.0

    while ( err > tol and  i < MaxIter ):

    # New uj vector
        uj_n[0]=uj1
        uj_n[1:N-1]=spsolve(AU ,  BQ*qj[1:N-1] +  Fj[1:N-1]+  b1*(1./24.*qja-m1*uj1)    )
        uj_n[N-1]=np.matmul([-1 ,2] ,uj_n[N-3:N-1] ) # Linear extrapolation
    
     
     # Left boundary extrapolated values of uj (cubic extrapolation)
        ujm1 = np.matmul([10 ,-20, 15, -4],uj_n[0:4] )
        uj00  = np.matmul([4  , -6,  4, -1],uj_n[0:4] )
    
     # Right boundary extrapolated values of uj (linear extrapolation)
        ujN1=np.matmul([-1, 2],uj_n[N-2:N] )
        ujN2=np.matmul([-2, 3],uj_n[N-2:N] )
  
    # Explicit derivative estimate
        Fj=KF*np.concatenate([np.array([ujm1]), np.array([uj00]), uj_n, np.array([ujN1]), np.array([ujN2])])
    
   # New qj vector
        qja=np.matmul (24./26. *((1+rj[0])/24.*np.array([1., 22., 1.])-1./dx**2*np.array([bj[0] ,-(aj[0]+bj[0]), aj[0]]) ),([ uj00 ,uj_n[0], uj_n[1] ]) ) -24./26.*Fj[0]
        qj_n[1:N-1]=(P1-P2*qja)
        qj_n[0] = 1./26.*np.matmul( np.array([5, -4, 1]),qj_n[1:4] ) + qja
        qj_n[N-1]=np.matmul(([-1, 2]),qj_n[N-3:N-1] )    # Linear extrapolation
    
    # Relative error
        err=(np.sum(np.abs(uj_n-uj))/np.sum(np.abs(uj_n)))

    # Actualization
        uj=beta1*uj_n+(1-beta1)*uj
        qj[0]=beta2*qj_n[0]+(1-beta2)*qj[0]
        qj[1:N]=qj_n[1:N]
        i=i+1
 
  #  if i >= MaxIter:
  #      print('WARNING : Maximum number of iterations reached')

    return uj, qj
