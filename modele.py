from DocomputGPP_sensiZ import *
import numpy as np
import matplotlib.pyplot as plt
import random
from randomfield_2 import *
import os
import __builtin__



#from numba import njit
    
    #Zprof_Perturb
#Zprof_Perturb: vecteur (200 points) de perturbation de la bathy dans l'intervalle [0 1]
# Zprof_Perturb=0.5-rand(200,1);
    
    
#@njit
def modele( RootFolder = None, tfin     = None, miter       = None, 
            rd_bathy   = None, rd_wave  = None, rd_wave_tot = None, 
            nstart     = None, nfps     = None, newdx       = None, 
            withnoise  = None, noisemin = None, noisemax    = None  ):

    read_bathy = False
    save_bathy = False
        
    # Generation Bathy Random
    success = False
    if rd_bathy : 
        while success == False :
            Zprof, success = genebathy_2()
    else : 
        Zprof=np.linspace(-10,2,num=200)
#  plt.plot(Zprof)
#    plt.show()    
    
    if read_bathy:
        fid=__builtin__.open(('Zprof_'+str(miter+1)+'.dat'),'r')
        toto=fid.read()
        Zprof=toto.strip().split("\t")
        Zprof=np.array([float(i) for i in Zprof])
        fid.close()

    np.save('Zprof',Zprof) 
    if save_bathy:
        fidZ=__builtin__.open('Zprof_'+str(miter+1)+'.dat','w')
        for idx in range(len(Zprof)):
          fidZ.write( '%6.8f \t' % Zprof[idx] )
        fidZ.close()
     
    # Time integration to compute eta
    t = DocomputGPP_sensiZ(RootFolder, tfin,miter, rd_wave, rd_wave_tot )
    
    # Post process bulk variables
    XStack, TimeStack, Bath = movieRESminInterne( nstart,tfin,nfps,1,RootFolder+'/results/calc0'+str(miter),RootFolder+'/', newdx )
 
    ratio=1
    TimeStackNoise = np.zeros( TimeStack.shape )

 
    if withnoise: 
    # Add noise
        noise = randomfield_2( TimeStack.shape, np.array([1,1]), np.array([100,10]), 80)
        level = random.uniform( noisemin, noisemax )   
        nn = (noise / 4. + 1) / 2.   #  -4<noise<4 => 0<nn<1
        nn = nn/np.max(nn)*level

#        ratio = np. round( 100.0*(len(ii) / len(np.asanyarray(nn)) ) )

        TimeStackNoise = TimeStack * nn
    else:
        nn=1.*TimeStackNoise
    
    
    # Save outputs
    if withnoise:
        np.savez( RootFolder+'/results/calc0'+str(miter)+'/timestack', XStack, Bath, TimeStack,TimeStackNoise ) 
    else:
        np.savez( RootFolder+'/results/calc0'+str(miter)+'/timestack', XStack, Bath, TimeStack )  

    return TimeStack, Zprof, nn
    
if __name__ == '__main__':
    pass


def movieRESminInterne(ti=None,tf=None,nfps=None,dum=None,carpeta=None,RootFolder=None,newdx=None):

# Arguments :
#  ti : initial time (s)
#  tf : final time (s)
#  nfps : number of frame per second for animation
#  dum = 1 to show movie
    #Loading computed data
    param = np.loadtxt(carpeta+'/param.dat')
    H0      = param[0]
    h0      = param[1]
    T0      = param[2]
    L0      = param[3]
    dx      = param[4]
    dt      = param[5]
    Ntot    = param[6]
    Tf      = param[7]
    ndt     = param[8]
    epsilon = param[9]
    kh0     = param[10]
    slope   = param[11]

    Xj= np.loadtxt(carpeta+'/Xj.dat')
    Fj= np.loadtxt(carpeta+'/Fj.dat')

    Time = np.loadtxt(carpeta+'/time.dat')
    Nwet = np.loadtxt(carpeta+'/Nwet.dat')

    Hjmat = np.loadtxt(carpeta+'/Hj.dat')
    NUjmat= np.loadtxt(carpeta+'/NUj.dat')

    mycmd='rm -f '+carpeta+'/*.dat '
    os.system(mycmd)

    Ntot  = NUjmat.shape[1]
    N0 = Nwet[0]
    a0 = H0/2.
    N  = N0

    #Animation
    dtf = 1./nfps
    j   = 0
    t0  = ti

    TimeStack = np.zeros( [ len(np.arange(ti+dtf,tf+dtf,dtf)), Ntot ] )
    Courant   = np.zeros( [ len(np.arange(ti+dtf,tf+dtf,dtf)), Ntot ] )
    Roller    = np.zeros( [ len(np.arange(ti+dtf,tf+dtf,dtf)), Ntot ] )

    

    for t in np.arange(ti+dtf,tf+dtf,dtf):
        Daux = abs(Time-t)
        Dist,tN = np.min(Daux), np.argmin(Daux)
        if Dist > 0:
            tN1 = tN-1
            tN2 = tN
        elif Dist < 0:
            tN1 = tN
            tN2 = tN+1
        elif Dist == 0:
            tN1 = tN
            tN2 = tN

        N1 = int(Nwet[tN1-1])
        N2 = int(Nwet[tN2-1])
        N  = min(N1,N2)
        t1 = Time[tN1-1]
        t2 = Time[tN2-1]
        Hj1 = Hjmat[tN1-1,0:N]
        Hj2 = Hjmat[tN2-1,0:N]
        NUj1 = NUjmat[tN1-1,0:N]
        NUj2 = NUjmat[tN2-1,0:N]

        if Dist != 0:
            Hj  = (t2-t)/(t2-t1)*Hj1 +(t-t1)/(t2-t1)*Hj2
            NUj = (t2-t)/(t2-t1)*NUj1+(t-t1)/(t2-t1)*NUj2    
        else:
            Hj  = Hj1
            NUj = NUj1
        
      
  #      NUj = NUj*(NUj>0)
  #      NUj = ( Hj[0:N]+Fj[0:N]+NUj ) *(NUj>0)
  #      ji = np.where(NUj<=0)
  #      NUj[ji] = ( Hj[ji]+Fj[ji] )
    
     # On remplie la matrice de niveau d'eau
        TimeStack[j,0:N] = Fj[0:N] + Hj[0:N]
    
     # remplie la matrice d ecourant
 #       Courant[j,0:N] = NUj[0:N]

     #On remplie la matrice de longueur de roller
 #       Roller[j,0:N] = NUj[0:N] - ( Fj[0:N]+Hj[0:N] )

        M = 0
        j = j+1

    Bath = Fj[0:N]
    aa, bb = TimeStack.shape
    XStack = Xj[0:bb]

    Zprof = np.load('Zprof.npy')
    # pas d espace en sortie : 2
    x = np.arange(0,len(Zprof)+newdx,newdx)

    TimeStack_new = np.zeros([aa,len(x)])
 #   Courant_new   = np.zeros([aa,len(x)])
#    Roller_new    = np.zeros([aa,len(x)])
    Bath_new      = np.zeros([aa,len(x)])
    for idt in range(aa):
        set_interp = interp1d( XStack[0:bb], TimeStack[idt,:], kind='linear', fill_value='extrapolate' )
        TimeStack_new[idt,:] = set_interp(x)
       
 #       set_interp = interp1d( XStack[0:bb], Courant[idt,:]  , kind='linear', fill_value='extrapolate' )
 #       Courant_new[idt,:] = set_interp(x)
 #       
 #       set_interp = interp1d( XStack[0:bb], Roller[idt,:]   , kind='linear', fill_value='extrapolate' )       
 #       Roller_new[idt,:] = set_interp(x)
       
        set_interp = interp1d( XStack[0:N], Bath, kind='linear', fill_value='extrapolate')
        Bath_new[idt,:] = set_interp(x)
#
    return x, TimeStack_new, Bath_new


def genebathy_2():

    X  = np.arange(0.,151)
    X1 = np.arange(-50.,151.)

    # interval number bars
    ib = np.array([0,5])
    b = random.random()
    bnb = int(round((ib[1]-ib[0])*b + ib[0]))   #number of bars

    # slope range
    ib = np.array([0.02, 0.05])
    b = random.random()
    sl=((ib[1]-ib[0])*b + ib[0])       #linear beach slope
    # bar amplitude range
    ib = np.array([0.25, 6])
    b = np.random.rand(bnb,1)
    bamp = sorted(((ib[1]-ib[0])*b+ ib[0]))

    # bar position range
    b = np.random.rand(bnb,1)
    ib = np.array([1*np.size(X)/5., 3*np.size(X)/5.])
    bx = sorted(np.round((ib[1]-ib[0])*b+ ib[0]).astype(int) ) 
 #   bx = [int(i) for i in bx]

    # bar width range
    ib = np.array([50., 600.])
    b = np.random.rand(bnb,1)
    bw = sorted(np.round((ib[1]-ib[0])*b + ib[0]).astype(int)  )
  #  bw = [int(i) for i in bw]

    z1 = -X*sl;    
    for ib in range (0,bnb) :
        bw[ib] = 3.*bx[ib];
        while z1[bx[ib]]+bamp[ib]*np.exp(  -((X[bx[ib]]-bx[ib]))**2/bw[ib] ) > z1[bx[ib]]/3.  :
            bamp[ib] = bamp[ib]/2.
        z1 = z1+bamp[ib]*np.exp( -((X-bx[ib])**2)/bw[ib] )

    mysucc=True
  #  mytest=z1[z1>0]
  #  if mytest.size > 0 : 
  #      print 'Generation bathy failed'
  #      mysucc=False

    z = -X1*sl 
    z[50::] = z1
    z[:] = z[::-1]
    z=z-0.2

    return z, mysucc
    
