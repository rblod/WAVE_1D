import numpy as np
from math import floor
#from numba import jit
# findposRL1.m

    
#@jit
def findposRL1(beta, *data):


    dx, dt ,uj0 ,hj0 ,g =  data

    # Water depth and depth-averaged velocity at location xR (departure point for the left going characteristic)
    p = int(floor(beta))
    b = beta - p
    uR0 = (1 - b)*uj0[p ] + b*uj0[p + 1]
    hR0 = (1 - b)*hj0[p ] + b*hj0[p + 1]
    err = dt*( uR0 - (g*hR0)** 0.5) + beta*dx 
    return err
