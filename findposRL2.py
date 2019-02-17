from math import floor
#from numba import jit
    

#@jit
def findposRL2(beta=None,dx=None,dt=None,uj0=None,hj0=None,hA1=None,uA1=None,g=None):

    # Water depth and depth-averaged velocity at location xR (departure point for right going characteristic)
    p = int(floor(beta))
    b = beta - p
    uR0 = (1 - b)*uj0[p] + b*uj0[p+1]
    hR0 = (1 - b)*hj0[p] + b*hj0[p+1]
    err = dt*(1./2.*(uR0 + uA1) - 1./2.*((g*hR0) ** 0.5 + (g*hA1)**0.5)  )  + beta*dx
    
    return err
