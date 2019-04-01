import numpy as np
from math import tan
#from numba import jit

#@jit
#@profile
def BrWaveIndex(t=None,kTb=None,PHIb=None,PHIf=None,hj=None,uj=None,fj=None,MatWav=None,kBrWaveOld=None,dx=None,dt=None,g=None,WDtol=None):

    # ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Arguments :
#  t : time variable (s)
#  kTb : coefficient for transitional time Tb=kTb*(h/g)^0.5
#  PHIb : critical breaker angle for iniciation of breaking (rad)
#  PHIf : saturated breaker angle (rad)
#  hj : cell face water depth values (m)
#  uj : cell face water velocity values (m)
#  fj : bottom coordinate (m)
#  MatWav : matrix array with wave information ([jC jT tanPHI])
#  kBrWaveOld : previous kBrWave array
#  dx : spatial grid resolution (m)
#  dt : time step
#  g : downward gravitational acceleration
#  WDtol : water depth tolerance
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# kBrWave = [(tb jb Tb jc jt tan(PHI) d c),k] : multidimensionnal array where tb is the time (s) when the breaking 
# was iniciated, jb is j-position of the breaking point, Tb is the transitional time, jc is the current j-position of the 
# k-wave crest, tan(PHI) is the associated front wave slope, d=0.5*(hC+hT) is the local average between crest 
# and trough water depths, c is the local wave celerity (m/s), and k is the wave index
# ----------------------------------------------------------------------------------------------------
    
    # ----------------------------------------------------------------------------------------------------
    N = hj.size
    
    # ----------------------------------------------------------------------------------------------------
    if kBrWaveOld.size == 0:
        kempty=True
        kBrWave = np.array([])
    else:
        kempty=False
    Nw=0
    if MatWav.size > 0:
        Nw=MatWav.shape[0]

    kBrWave=np.array([])

    if ( Nw != 0 and kempty == True):
        k = 0
        for i in range(Nw):
            jc = int(MatWav[i,0])
            jt = int(MatWav[i,1])
            tanPHI = MatWav[i,2]
            d = MatWav[i,3]
            c = (g*d)** 0.5

            if ( tanPHI > tan(PHIb) and uj[jc] > 0 and hj[jt] > WDtol ) :
                tb = t
                jb = jc
                Tb = kTb *  (d/g)** 0.5
                if k ==0 :
                    kBrWave=np.zeros([8,1])
                    kBrWave[0:8,0]=np.array( [tb, jb, Tb, jc, jt, tanPHI, d, c] )
                else:
                    toto=np.zeros([8,1])
                    toto[:,0]= np.array( [tb, jb, Tb, jc, jt, tanPHI, d, c])
                    kBrWave = np.append(kBrWave,toto,axis=1 )
                k = k + 1

    elif ( Nw !=0 and kempty == False ) :
        k = 0
        jbr = kBrWaveOld[3,:]
        for i in range(1,Nw):
            jc  = int(MatWav[i,0])
            jt  = int(MatWav[i,1])
            jt1 = int(MatWav[i-1,1])
            tanPHI = MatWav[i,2]
            d = MatWav[i,3]
            c = (g*d) ** 0.5
            kj=np.array([])
            myall= np.where( (jbr > jt1) & (jbr < jt)  )[0]
            if myall.size>0:
                 kj=min(myall)
            if ( kj.size > 0  and  tanPHI > tan(PHIf) and uj[jc] > 0  and hj[jt] > WDtol ):
                toto=np.zeros([3,1])
                toto[:,0]=kBrWaveOld[0:3,kj]
                titi=np.zeros([5,1])
                titi[:,0] = np.array([jc, jt, tanPHI, d, c])
                tata=np.zeros([8,1])
                tata[0:3,0]=toto[:,0]
                tata[3:8,0]=titi[:,0]

                if k ==0 :
                    kBrWave=np.zeros([8,1])
                    kBrWave[0:8,k]= tata[:,0]
                else   : 
                    kBrWave = np.append(kBrWave,tata,axis=1 )
                k = k + 1

            elif ( kj.size == 0 and  tanPHI > tan(PHIb) and uj[jc] > 0 and hj[jt] > WDtol ):
                print 'in part 2'
                tb = t
                jb = jc
                Tb = kTb*(d/g)** 0.5
                tata=np.zeros([8,1])
                tata[:,0]=np.array( [tb, jb, Tb, jc, jt, tanPHI, d, c] )
                if k ==0 :
                    kBrWave=np.zeros([8,1])
                    kBrWave[0:8,k]= tata[:,0]
                else:    
                    kBrWave= np.append(kBrWave,tata,axis=1 )
                k = k + 1
    elif (Nw == 0):
        kBrWave=np.array([])
    kBrWaveOld= np.array([])

    return kBrWave
