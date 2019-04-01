import numpy as np
from numpy import dot, multiply
from math import exp,log,tan,ceil
#from numba import jit

    
#@jit
def EddyViscosity(t=None,PHIb=None,PHIf=None,beta=None,alfab=None,alfaf=None,gamab=None,gamaf=None,kapb=None,kapf=None,kBrWave=None,Xj=None,hj=None,fj=None,dx=None,g=None,WDtol=None):

    # ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Arguments :
#  t : time variable (s)
#  PHIb : critical breaker angle for iniciation of breaking (rad)
#  PHIf : saturated breaker angle (rad)
#  beta : Cointe and Tulin (1994) parameter
#  alfab : order one parameter for energy dissipation at breaking point
#  alfaf : order one parameter for energy dissipation at the end of transition zone
#  gamab : breaker index at the breaking point (wave height over MWL)
#  gamaf : saturated breaker index
#  kapb : ratio Kh/Khu near the breaking point
#  kapf : ratio Kh/Khu in the inner surf zone
#  kBrWave : wave array [(tb jb Tb jc tan(PHI) d c),k]
#  Xj : cell face x-coordinates (m)
#  hj : cell face water depth values (m)
#  fj : cell face bottom bathymetry (m)
#  dx : spatial grid resolution (m) 
#  g : downward gravitational acceleration (m/s^2)
#  WDtol : tolerance for water depth (m)
    
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ------ Spatial distribution of eddy viscosity coefficients in breaking wave parametrization --------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
    
    N = hj.size
    Nu1j = np.zeros(N)
    Nu2j = np.zeros(N)
    K = 3.*exp(- 1) - 1
    i=0
    if kBrWave.ndim <=1:
        kn=0
        jB=np.array([0])
    else:
        kn=np.shape(kBrWave)[1]
        jB=np.zeros(kn)
  
    
# ----------------------------------------------------------------------------------------------------
# -------------------------------------- Cienfuegos et al. (2005) ------------------------------------
# ----------------------------------------------------------------------------------------------------
    
    for k in range(kn):
        kDAT = kBrWave[0:8,k]
        tb = kDAT[0]
        jb = int(kDAT[1])
        Tb = kDAT[2]
        jc = int(kDAT[3])
        jt = int(kDAT[4])
        tanPHI = kDAT[5]
        d = kDAT[6]
        c = kDAT[7]
    #    if c==0:
    #        return
        PHIr = PHIf + (PHIb - PHIf) * exp(-log(2)*(t-tb)/Tb)
        gamar = gamaf + (gamab-gamaf)*exp(-log(2)*(t-tb)/Tb)    # Breaker index evolution
        alfar = alfaf + (alfab-alfaf)*exp(-log(2)*(t-tb)/Tb)    #  Order one coefficient for energy dissipation evolution
        er = max(beta**2/(2.*(1-beta**2))*(1-gamar)*d,WDtol)    # Total roller height
        lr=er/tan(PHIr)    #Total roller length
        Jr1=jc             # j-coordinate for the location of the first "active" node
        Jr2 = min(Jr1+max(int(ceil(lr/dx)),2),N-1)    # j-coordinate for the last "active" node
        lr = min(Xj[Jr2]-Xj[Jr1],Xj[jt]-Xj[Jr1])
        K1K2 = kapf+(kapb-kapf)*exp(-log(2)*(t-tb)/Tb)    #Diffusivity weighting coefficient evolution 
        K2 = alfar/(1.+K1K2)/tan(PHIr)                     #Momentum eddy viscosity coefficient
        K1 = K1K2*K2                                     # Continuity eddy viscosity coefficient
        jB[i] = jb #Breaking nodal point
        i = i + 1
        Xr = Xj[Jr1:Jr2+1]- Xj[Jr1]
        Nu1j[Jr1:Jr2+1] = -K1*c*d*np.exp(Xr/lr-1.)*((Xr/lr-1.)+(Xr/lr-1.)**2) #Local eddy viscosity for continuity equation
        Nu2j[Jr1:Jr2+1] = -K2*c*d*np.exp(Xr/lr-1.)*((Xr/lr-1.)+(Xr/lr-1.)**2) # Local eddy viscosity for momentum equation    
     #   if k == kn-1 :
     #       print Xr
     #       print K1
     #       print Nu1j[Jr1:Jr2+1]
     #       print lr    


    jB1=int(max(np.min(jB),0) )
    xb=Xj[jB1]
    Nu1j = multiply((Nu1j > 0), Nu1j)
    Nu2j = multiply((Nu2j > 0), Nu2j)
    
  #  print 'titi'
  #  print Nu1j

    #print 'end iti'
    
    return Nu1j,Nu2j,xb
