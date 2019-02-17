import numpy as np
from math import log 
from scipy.interpolate   import interp1d
#from numba import jit

    
#@jit
def DataManip(t=None,h0=None,DAT=None):

    time = DAT[:,0]
    eta  = DAT[:,1]
    Tini = 3*(9.81*h0)**0.5
 #   A=np.ones(time)
 #   if t < Tini:
 #       J=np.where(time < Tini)
 #       A[J]=log(time(J) / Tini *(exp(1.) - 1.) + 1)
 #       
    set_interp = interp1d( time, eta, kind='linear',fill_value='extrapolate' )
    E=set_interp(t)
    h=h0 + E
 #   return h,E    
    return h
