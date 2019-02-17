from numpy import dot, multiply
from numba import njit
    
#@njit
def FluxMomentum(qj=None,hj=None,fj=None,uj=None,Dxhj=None,Dxfj=None,Dxuj=None,Dxxhj=None,Dxxuj=None,Dxxfj=None,g=None,bD=None,alfa=None,WDtol=None):
 
    N=len(hj)       
    FMj=qj*uj+g*(hj+fj)-1/2.*uj**2+bD*(hj[0:N]>WDtol)*(-1./2.*Dxfj**2*uj**2+(Dxfj*uj-1./2.*hj*Dxuj)*hj*Dxuj-alfa*fj**2*(Dxuj**2+uj*Dxxuj+g*(Dxxhj+Dxxfj)))
    
    return FMj
