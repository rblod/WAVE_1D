import numpy as np
from scipy.signal import butter    
from scipy.signal import filtfilt 
#from numba import jit
from explicitFILT0 import *


#@jit
#@profile
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
  #  print 'in wave prop N = ', N
    
    etaj = hj + fj

    Dxetaj=explicitFILT0( Dxhj + Dxfj )
    
    dx = Xj[1] - Xj[0]
    jc=np.array([])
    jt=np.array([])
    MatWav=np.array([])

    # Mean Water Level
    fNorm =0.008
    b, a = butter(1,fNorm,'low')
    etajm = filtfilt(b, a, etaj,padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    E = etaj - etajm

    jzero=np.zeros([])
    c=np.zeros(E.shape)
    # Estimates for zero up-crossing
    j1 = np.where(E[0:N-2] < 0 )
    j2 = np.where(E[1:N-1] > 0 )
    inter = np.intersect1d(j1, j2)
    c[inter] = 1
    j3=np.nonzero(c)
  #  print 'j3 = ', j3

    if (j3[0].size )> 0:
        jzero=j3[0]+1
        jzero = np.concatenate( [ jzero,np.array([N-1])] )
    else:
        jzero = np.array([N-1])
  #  print 'yo', len(j3), jzero

    if ( jzero.size > 0 and (len(jzero) > 2) ):
        k=0
     #   jc=np.zeros([])
     #   jt=np.zeros([])
     #   MatWav=np.zeros([])

    #    print 'jzero = '
    #    print jzero
        for i in range(0,(len(jzero)- 1)):
            Emax,jcc = np.max(E[jzero[i]:jzero[i+1]+1] ), np.argmax(E[jzero[i]:jzero[i+1]+1] )
            Emin,jtt = np.min(E[jzero[i]:jzero[i+1]+1] ), np.argmin(E[jzero[i]:jzero[i+1]+1] )
      #      print 'jc,jt'
      #      print jcc,jtt

            jc1=jzero[i] + jcc 
            jt1=jzero[i] + jtt 
            if ( jt1 > jc1  and  hj[jt1] > WDtol ):
                if fj[jc1] < - dx:
                    d = -fj[jc1]
                else:
                    d = dx
                tanPHI = - np.min(Dxetaj[jc1:jt1+1])
                if k==0:
                    MatWav=np.zeros([1,4])
                    jc=np.zeros([1])
                    jt=np.zeros([1])

                    MatWav[k,0]=jc1
                    MatWav[k,1]=jt1
                    MatWav[k,2]=tanPHI
                    MatWav[k,3]=d
                    jc[k]=jc1
                    jt[k]=jt1
                else:
                    toto=np.zeros([1,4])
                    toto[0,:]=np.array([jc1,jt1,tanPHI,d])
                    MatWav=np.append(MatWav,toto,axis=0)
                    jc=np.append(jc,jc1)
                    jt=np.append(jt,jt1)
                k = k + 1
    #        if jt1==1: 
    #            print  jtt,jzero[i] ,jt1
    #            return 
   # print 'in waveprop'
   # print MatWav
   # print 'en waveprop'

    return jc,jt,MatWav
