import numpy as np
from scipy.signal import butter    
from scipy.signal import filtfilt 
#from numba import jit
from explicitFILT0 import *


#@jit
def WaveProp(Xj=None, hj=None, fj=None, Dxhj=None,Dxfj=None,WDtol=None):
    # Estimate of j-position indexes of wave crest and trough
    
    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # Arguments :
    #  Xj : cell face x-coordinates (m)
    #  hj : cell face water depth values (m)
    #  fj : cell face bottom bathymetry (m)
    #  Dxhj : 1st x-derivative of water depth
    #  Dxfj : 1st x-derivative of bottom bathymetry
    
    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    N = hj.size
    
    etaj = hj + fj
    
    Dxetaj=explicitFILT0( Dxhj + Dxfj )
    
    dx = Xj[1] - Xj[0]
    jc=np.array([])
    jt=np.array([])
    MatWav=np.array([])

    # Mean Water Level
    fNorm =0.008
    b, a = butter(1,fNorm,'low')
    etajm = filtfilt(b, a, etaj)
    E = etaj - etajm
   # print 'ETA', 

    # Estimates for zero up-crossing
    j1 = np.where(E[0:N-2] < 0 )
    j2 = np.where(E[1:N-1] > 0 )
    j3 = j1 and  j2  
    jzero=j3[0]+1
    jzero = np.concatenate( [ jzero,np.array([N])] )

    if ( jzero.size > 0 and (len(jzero) > 2) ):
        k=0
        jc=np.zeros(len(jzero)- 1)
        jt=np.zeros(len(jzero)- 1)
        MatWav=np.zeros([len(jzero)- 1,4])

        for i in range(0,(len(jzero)- 1)):
            Emax,jcc = np.max(E[jzero[i]:jzero[i+1]+1] ), np.argmax(E[jzero[i]:jzero[i+1]+1] )
            Emin,jtt = np.min(E[jzero[i]:jzero[i+1]+1] ), np.argmin(E[jzero[i]:jzero[i+1]+1] )
            jc1=jzero[i] + jcc - 1
            jt1=jzero[i] + jtt - 1
            if ( jt1 > jc1  and  hj[jt1] > WDtol ):
                if fj[jc1] < - dx:
                    d = -fj[jc1]
                else:
                    d = dx
                tanPHI = - np.min(Dxetaj[jc1:jt1+1])
                MatWav[k,0]=jc1
                MatWav[k,1]=jt1
                MatWav[k,2]=tanPHI
                MatWav[k,3]=d
                jc[k]=jc1
                jt[k]=jt1
                k = k + 1
   # print 'in waveprop'
   # print MatWav
   # print 'en waveprop'
    return jc,jt,MatWav
