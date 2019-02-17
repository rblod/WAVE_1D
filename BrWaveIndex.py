import numpy as np
from math import tan
#from numba import jit

#@jit
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
    N = len(hj)
    
    # ----------------------------------------------------------------------------------------------------
    kempty = len(kBrWaveOld) == 0
    Nw=0
#    print MatWav
    if len(MatWav) > 0:
        Nw=MatWav.shape[0]

#    print 'tttss ', kBrWaveOld
 #   print 'tttttt ', Nw
#    print 'ssstt ', MatWav


    if ( Nw != 0 and kempty == 1):
        kBrWave = np.zeros([8,Nw])
        k = 0
        for i in range(Nw):
            jc = MatWav[i,0]
            jt = MatWav[i,1]
            tanPHI = MatWav[i,2]
            d = MatWav[i,3]
            c = (g*d)** 0.5
            if ( tanPHI > tan(PHIb) and uj[jc] > 0 and hj[jt] > WDtol ) :
                tb = t
                jb = jc
                Tb = kTb *  (d/g)** 0.5 
                kBrWave[0:8,k] = np.array( [tb, jb, Tb, jc, jt, tanPHI, d, c] )
                k = k + 1
    else:
        kBrWave = np.zeros([8,Nw])
        if ( Nw != 0 and kempty != 1) :
            k = 0
            jbr = kBrWaveOld[3,:]
            for i in range(1,Nw):
                jc  = MatWav[i,0]
                jt  = MatWav[i,1]
                jt1 = MatWav[i-1,1]
                tanPHI = MatWav[i,2]
                d = MatWav[i,3]
                c = (g*d) ** 0.5
            #    kj= np.min( np.where( jbr > jt1 and jbr < jt) )
                my1=np.where( jbr > jt1 )[0]
                my2=np.where( jbr < jt  )[0]
                min1=np.array([])
                min2=np.array([])
                kj=np.array([])
                if (my1.size > 0):
                    min1=np.min(my1)
                if (my2.size > 0):
                    min2=np.min(my2)
                if ( min1.size>0 and min2.size )>0:
                    kj=min(min1,min2)
                elif min1.size >0:
                    kj=min1
                elif min2.size >0:
                    kj=min2    

       #         kj= (np.min( np.where( jbr > jt1 ), np.min( np.where( jbr < jt))))

                if ( kj.size > 0  and  tanPHI > tan(PHIf) and uj[jc] > 0  and hj[jt] > WDtol ):
                    toto=np.zeros(3)
                    toto[:]=kBrWaveOld[0:3,kj]
          #          print  type(jc), type(jt), type(tanPHI), type(d), type(c)
                    titi =np.array([jc, jt, tanPHI, d, c])
                    kBrWave[0:8,k] = np.concatenate( [toto , titi ] )
                    k = k + 1
                else:
                    if ( kj.size == 0 and  tanPHI > tan(PHIb) and uj[jc] > 0 and hj[jt] > WDtol ):
                        tb = t
                        jb = jc
                        Tb = kTb*(d/g)** 0.5
                        kBrWave[0:8,k] = np.concatenate( [tb, jb, Tb, jc, jt, tanPHI, d, c] )
                        k = k + 1
        else:
            if (Nw == 0):
                kBrWave=np.array([])
    
    return kBrWave