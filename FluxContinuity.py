import numpy as np
from numba import njit
    
#@njit
#@profile
def FluxContinuity(hj=None,uj=None,Dxhj=None,Dxfj=None,Nuj=None):

    # ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# -------------------- Flux function estimate for continuity equation --------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Arguments :
#  hj : cell face water depth values (m)
#  uj : cell face depth averaged velocity values (m/s)
#  Dxhj : 1st x-derivative of water depth
#  Dxfj : 1st x-derivative of bottom bathymetry
#  Nuj : celle face eddy viscosity values for breaking paramterization
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
    N=len(hj)
    FCj = np.zeros(N)
    FCj = hj*uj - Nuj*Dxhj
    return FCj
