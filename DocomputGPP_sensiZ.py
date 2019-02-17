from os import *
from FVRK4GPP_sensiZ import *
import numpy as np
import os
#from numba import njit

#@njit
def DocomputGPP_sensiZ( RootFolder=None, tend=None, miter=None, rd_wave=None, rd_wave_tot=None ):

    t=0
    Cr=1

    #Boucle qui reduit Cr si la simu plante
    try:
        STO = 1
        mycmd = 'mkdir -p '+RootFolder+'/results/calc0'+str(miter)
        os.system(mycmd)
        carpetaN = RootFolder+'/results/calc0'+str(miter)
        carpetaO = RootFolder+'/results/calc0'+str(miter)
        tini = t
        mycmd ='cd '+RootFolder
        os.system(mycmd)
        t = FVRK4GPP_sensiZ(tini, tend, Cr, STO, carpetaN, carpetaO, RootFolder, rd_wave, rd_wave_tot)
        if t < tend:
            print 'AAAAAAA AAAAA'
            Cr = 0.1
            t = 0
            STO = 1
            mycmd = 'mkdir -p '+RootFolder+'/results/calc0'+str(miter)
            os.system(mycmd)
            carpetaN = RootFolder+'/results/calc0'+str(miter)
            carpetaO = RootFolder+'/results/calc0'+str(miter)
            t = FVRK4GPP_sensiZ(tini, tend, Cr, STO, carpetaN, carpetaO, RootFolder, rd_wave, rd_wave_tot)
    finally:
        pass
    return t
    
    
    
