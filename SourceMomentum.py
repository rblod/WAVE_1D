import numpy as np
from numba import njit
    
#@njit
def SourceMomentum(hi=None,hj=None,fi=None,fj=None,uj=None,Dxhj=None,Dxfj=None,Dxfi=None,Dxuj=None,Nuj=None,Xi=None,g=None,bD=None,alfa=None,fricFact=None,Cfi=None,WDtol=None):

    # ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ------------------------------ Source terms for momemtum equation ----------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Arguments :
#  hi : volume centered values (i) for water depth (m)
#  hj : cell face water depth values (m)
#  fi : volume centered bottom bathymetry (m)
#  fj : cell face bottom bathymetry (m)
#  uj : cell face depth averaged velocity values (m/s)
#  Dxhj : 1st x-derivative of water depth
#  Dxfj : 1st x-derivative of bottom bathymetry (j values)
#  Dxfi : 1st x-derivative of bottom bathymetry (i values)
#  Dxuj : 1st x -derivative of depth averaged velocity
#  Nuj : celle face eddy viscosity values for breaking paramterization
#  g : downward gravity acceleration (m/s^2)
#  bD : parameter for dispersive terms (active == 1 ; neglect == 0)
#  alfa : linear dispersion correction parameter
#  fricFact : == 0 for Grant and Madsen (1979), == 1 for Manning resistance formula 
#  Cfi : friction coefficient at i nodal points 
#  WDtol : mimimum water depth 
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
    N=len(hj)
    if fricFact == 0:
        Cfi[0:N-1]=Cfi[0:N-1]
    else:
        if fricFact == 1:
            Cfi=2*g*hi*Cfi**2/hi**(4./3.)
    
    
    S1i = Nuj[1:N]*(hj[1:N]*Dxuj[1:N]+Dxhj[1:N]*uj[1:N])-Nuj[0:N-1]*(hj[0:N-1]*Dxuj[0:N-1]+Dxhj[0:N-1]*uj[0:N-1])

    S2i = -2.*bD*alfa*(hi[0:N-1]>WDtol) *((uj[1:N] *Dxuj[1:N]+g*(Dxhj[1:N]+Dxfj[1:N]))-(uj[0:N-1] *Dxuj[0:N-1]+g*(Dxhj[0:N-1]+Dxfj[0:N-1])))
   
   # S2i = -2 *bD*alfa*(hi(1:N-1)>WDtol).*((uj(2:N).*Dxuj(2:N)+g*(Dxhj(2:N)+Dxfj(2:N)))-(uj(1:N-1).*Dxuj(1:N-1)+g*(Dxhj(1:N-1)+Dxfj(1:N-1))));

    S3i = -1./2.*Cfi*1./2.*(uj[0:N-1]+uj[1:N])*1./2.*np.abs(uj[0:N-1]+uj[1:N])
    
  #  print hi[150::]
  #  print uj[150::]
  #  print Dxuj[150::]
  #  print Dxhj[150::]
  #  print Dxfj[150::]
   # toto= (uj[1:N] *Dxuj[1:N]+g*(Dxhj[1:N]+Dxfj[1:N]))
  #  toto= (Dxfj[1:N] ) 
  #  print   toto[150::]



    SMi=1./hi[0:N-1]*(hi[0:N-1]>WDtol)*(S1i+S3i)+fi[0:N-1]*Dxfi[0:N-1]*S2i
    #print SMi[150::] 

    return SMi
