import numpy as np
import os
from scipy.interpolate   import interp1d
from scipy.sparse.linalg import spsolve
from scipy.optimize import fsolve, root
from math  import pi, floor
from numpy  import random
#from numba import jit

from explicitFILT0    import *
from implicitFILT8    import *
from InterpItoJ       import *
from jonswap_blw      import *
from Comp_1st_Mat     import *
from Cell_J_Mat       import *
from Deriv1j          import *
from deriv1i          import *
from Comp_2nd_Mat     import *
from Deriv2j          import *
from opfiles          import *
from VolIntMat        import *
from CellFaceInt      import *
from WaveProp         import * 
from BrWaveIndex      import *
from EddyViscosity    import *
from FluxContinuity   import *
from FluxMomentum     import * 
from SourceMomentum   import *
from DataManip        import *
from opfiles          import *
from findposRL1       import *
from findposRL2       import *
from DeffCorrU        import *

#@jit
def FVRK4GPP_sensiZ(Ti=None, Tf=None, Cr=None, STO=None, carpetaN=None, carpetaO=None, RootFolder=None, rd_wave=None, rd_wave_tot=None,miter=None):
    
    # ----------------------------------------------------------------------------------------------------
    # --------------------------------------- Main Routine -----------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # Method :
    # Numerical Method solving the Serre Equations for uneven bottoms
    # 4th order finite volume for spatial discretization
    # 4th order Runge-Kutta method for time stepping
    # Modified equations from Seabra-Santos et al. (1987)
    # Retaining terms of order O(sigma^2) and Pade [2,2] linear dispersion correction based on Madsen et al. (1991)
    #-
    # History
    # 4/02/2019 : Rachid Benshila LEGOS-CNRS    , Python version 
    # 4/01/2010 : Rodrigo Cienfuegos C. LEGI-PUC, Initial version
    # ----------------------------------------------------------------------------------------------------
    
#RB  lastwarn('')
    # ----------------------------------------------------------------------------------------------------
    # ---------------------------- Problem definition and parameters -------------------------------------
    # ----------------------------------------------------------------------------------------------------
    
    # ----------------------------------------------------------------------------------------------------
    # Downward gravitational acceleration (m\s^2)
    # ----------------------------------------------------------------------------------------------------
    #-
    g = 9.81

    # ----------------------------------------------------------------------------------------------------
    #-
    read_forcing = False
    read_bathy=False
    save_forcing = False
    full_output = False
    
    if read_bathy:
        fid=open(('Zprof_'+str(miter+1)+'.dat'),'r')
        toto=fid.read()
        Zprof=(toto.strip().split("\t"))
        Zprof=np.array([float(i) for i in Zprof])
        fid.close()
    else:
        Zprof = np.load( RootFolder+'/Zprof.npy' )


    # ----------------------------------------------------------------------------------------------------
    #
    # Incident wave field
    #Profondeur d'eau au large
    h0 = abs(Zprof[0])
    H0 = 0.5    
    a0 = H0/.2
    T0 = 10
    
    c0 = (g*h0)** 0.5    
    L0 = c0*T0    
    k0 = 2*pi/L0    
    kh0 = k0*h0    
    offset = 0
        
    # ----------------------------------------------------------------------------------------------------
    # Wave breaking parametrization
    # ----------------------------------------------------------------------------------------------------
    #-
    PHIb = 28.*pi / 180.    
    PHIf =  8.*pi / 180.    
    PHI1 = 13.*pi / 180.    
    PHI2 = 13.*pi / 180.    
    gamab = 0.8    
    gamaf = 0.8   
    kap1 = 0.1    
    kap2 = 0.1    
    alfab = 5.0    
    alfaf = 5.0   
    kTb = 5.0    
    bCointe = 0.796
        
    # ----------------------------------------------------------------------------------------------------
    # Numerical discretization
    # ----------------------------------------------------------------------------------------------------
    #-
    ro = 0.1    
    dx = ro*h0    
    dt = Cr/(g*h0)**0.5*dx    
    WDtol = 0.0001
        
    # ----------------------------------------------------------------------------------------------------
    # Linear dispersion correction (Madsen et al., 1991)
    # ----------------------------------------------------------------------------------------------------
    #-
    alfa = 1./15.
        
    # ----------------------------------------------------------------------------------------------------
    # Friction coefficient
    # ----------------------------------------------------------------------------------------------------
    #-
    fricFact = 0    
    Cf = 0.015
    
    
    # ----------------------------------------------------------------------------------------------------
    # Filtering Wet domain
    # ----------------------------------------------------------------------------------------------------
    #-
    FilterD1 = 1   
    FilterD2 = 1  
    NSZ = 15    
    NIZ = 15    
    NFT = 0    
    alfaF = 0.4
    
    # ----------------------------------------------------------------------------------------------------
    # Parameter values for Picard iterations in deferred-correction approach
    # ----------------------------------------------------------------------------------------------------
    #-
    MaxIter = 10    
    tol = 1e-05
    
    # ----------------------------------------------------------------------------------------------------
    # Dispersion terms (betaD=1 : Activate dispersif terms ; betaD=0 : Neglect dispersif terms)
    # ----------------------------------------------------------------------------------------------------
    #-
    betaD = 1

    # ----------------------------------------------------------------------------------------------------
    # Bottom bathymetry and physical domain (wet & dry)
    # ----------------------------------------------------------------------------------------------------
    #-
    slope = 0.02
    
    xf = np.arange( 1, len(Zprof)+1 )
    zf = Zprof
    zf[0:3] = zf[0]

    Xlong = max(xf)
    Xj = np.arange( xf[0], Xlong, dx )
    Xi = np.arange( xf[0] + dx/2., Xlong-dx/2., dx )
    
    set_interp = interp1d( xf, zf, kind='linear', fill_value='extrapolate' )
    fi = set_interp( Xi )
    fi = explicitFILT0( fi )

    set_interp = interp1d( Xi, fi, kind='linear', fill_value='extrapolate' )
    fj = set_interp( Xj[len(Xj)-1] )  
    fj = InterpItoJ( fi, zf[0], fj )
    
    Ntot = len(Xj)
    
    N= min ( np.min( np.where(fj > 0) ), np.max( np.where(fj <= 0) ) )   #python index
    
    Next=N + NSZ      #python index
    
    Nint=N - NIZ      #python index
    
    # ----------------------------------------------------------------------------------------------------
    # Wave forcing
    # ----------------------------------------------------------------------------------------------------
    #-
    
    # load([RootFolder,'/serie_temp_jonswap_sensi_terrasse_slope']);
# ii=find(etahf(1:end-1)<=0&etahf(2:end)>0);
# etahf=etahf(ii(1):end);tti=tti(ii(1):end);tti=tti-min(tti);
    
    if rd_wave_tot:
        rd_wave = True
        Tp1 =random.uniform(6.,15.) 
        H0=50
        while H0 > abs(Zprof[0])/4.:
            H0  = random.uniform(0.2,hmax)
    else:    
        Tp1 = 10.
        H0 = 0.5    
 
    a1  = H0
    a0 = H0/2.
    epsilon=a0/h0 #Typical nonlinear parameter

    if not read_forcing: 
        ff,SSm,tt,etahf,eta,etaLB = jonswap_blw( Tf*2, (1./dt), 1./Tp1, a1, 100, 3.3, -Zprof[0], rd_wave )
  

    if read_forcing :
        fid=open('eta_'+str(miter+1)+'.dat','r')
        toto=fid.read()
        etahf=(toto.strip().split("\t"))
        etahf=np.array([float(i) for i in etahf])
        fid.close()

        fid=open('tt_'+str(miter+1)+'.dat','r')
        toto=fid.read()
        tt=(toto.strip().split("\t"))
        tt=np.array([float(i) for i in tt])
        fid.close()

    if save_forcing:        
        fideta=open('eta_'+str(miter+1)+'.dat','w')
        for idx in range(len(etahf)):
            fideta.write( '%6.8f \t' % etahf[idx] )
        fideta.close()
        fidtt=open('tt_'+str(miter+1)+'.dat','w')
        for idx in range(len(etahf)):
            fidtt.write( '%6.8f \t' % tt[idx] )
        fidtt.close()
        np.save('etahf',etahf) 
        np.save('tt',tt) 
        fidXDS=open('Cr_'+str(miter+1)+'.dat','w')  
        fidXDS.write( '%6.8f \t' % Cr )
        fidXDS.close()


 
    DAT = np.zeros( [len(tt), 2] )
    DAT[:,0] = tt
    DAT[:,1] = etahf
    Cfi = Cf*np.ones( [Ntot] )
        
    # ----------------------------------------------------------------------------------------------------
    # ---------------------------- Output files with results and parameters ------------------------------ 
    # ----------------------------------------------------------------------------------------------------
    #-
    # Rows = spatial coordinate
    # Columns = temporal coordinate
    #-
    fidParam,fidXj,fidXi,fidFj,fidFxj,fidFxxj,fidHj,fidHi,fidUj,fidUi,fidUxj,fidUxxj,fidNUj,fidTime,fidNwet,fidXb,fidHout = opfiles(carpetaN,RootFolder )
    
    # ----------------------------------------------------------------------------------------------------
    # --------------------------------- Constant Matrix Generation ---------------------------------------
    # ----------------------------------------------------------------------------------------------------
    #-
    ADxj  = Comp_1st_Mat( Ntot   )    
    ADxi  = Comp_1st_Mat( Ntot-1 )     
    ADxxj = Comp_2nd_Mat( Ntot   )    
    AD    = Comp_1st_Mat( Ntot-2 )    
    ADi   = Comp_1st_Mat( Ntot-3 )    
    BQ    =   Cell_J_Mat( Ntot )
    
    # ----------------------------------------------------------------------------------------------------
    # ---------------------------- Bottom's derivative estimate ------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    #-
    Dxfj =  Deriv1j( fj, dx, ADxj )
    Dxfi =  deriv1i( fi, fj[0],fj[Ntot-1], dx, ADxi )
    Dxxfj = Deriv2j( fj, dx, ADxxj )
    

    for idx in range(len(Xj)):
        fidXj.write  ( '%6.4f \t' % Xj[idx]   )
        fidFj.write  ( '%6.4f \t' % fj[idx]   )
        fidFxj.write ( '%6.4f \t' % Dxfj[idx] )
    for idx in range(len(Xi)):
        fidXi.write  ( '%6.4f \t' % Xi[idx]   )

#    np.savetxt( carpetaN+'/Xj.dat'  , Xj   ,   fmt='%6.4f', delimiter='\t' )
#    np.savetxt( carpetaN+'/Xi.dat'  , Xi   ,   fmt='%6.4f', delimiter='\t' )
#    np.savetxt( carpetaN+'/Fj.dat'  , fj   ,   fmt='%6.4f', delimiter='\t' )
#    np.savetxt( carpetaN+'/Fxj.dat' , Dxfj ,  fmt='%6.4f', delimiter='\t' )
#    np.savetxt( carpetaN+'/Fxxj.dat', Dxxfj, fmt='%6.4f', delimiter='\t' )
    
    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # ---------------------------------- Initial Conditions ----------------------------------------------
    
    # ----------------------------------------------------------------------------------------------------
    # Initial conditions for water depth h and depth-averaged velocity u at cell faces (j nodes)
    # ----------------------------------------------------------------------------------------------------
    #-
    t0 = Ti

    Nu1j = np.zeros([Ntot])
    Nu2j = np.zeros([Ntot])
    Dxhj = np.zeros([Ntot])
    Dxxhj = np.zeros([Ntot])
    Dxuj = np.zeros([Ntot])
    Dxxuj = np.zeros([Ntot])
    FCj = np.zeros([Ntot])
    FCi = np.zeros([Ntot-1])
    FMj = np.zeros([Ntot])
    FMi = np.zeros([Ntot-1])
    SMi = np.zeros([Ntot-1])
    DtHi = np.zeros([Ntot-1])
    DtQi = np.zeros([Ntot-1])
    KQ1 = np.zeros([Ntot-1])
    kH1 = np.zeros([Ntot-1])
    KQ2 = np.zeros([Ntot-1])
    kH2 = np.zeros([Ntot-1])
    KQ3 = np.zeros([Ntot-1])
    kH3 = np.zeros([Ntot-1])
    KQ4 = np.zeros([Ntot-1])
    kH3 = np.zeros([Ntot-1])
   

    if t0 == 0:
        hj = -fj[0:Ntot]
        uj = np.zeros( [Ntot] )
        hi = -fi[0:Ntot-1]
        set_interp = interp1d( Xj, uj, kind='cubic' )
        ui = set_interp(Xi)
        qj = np.zeros( [Ntot] )
    else:
        hj, uj, hi, ui, t0 = readLastLine( carpetaO, RootFolder )
    aa=len(hj)
    bb=len(uj)
        
    # ----------------------------------------------------------------------------------------------------
    # Initial condition for auxiliary variable q at cell faces (j nodes)
    # ----------------------------------------------------------------------------------------------------
    #- 

    #-   
    # Cell-averaged values for depth-averaged velocity at j cells
    #-
    Uj = np.zeros([Ntot])
    Uj[0] = 2./3.*uj[0] + 1./6.*( 5*ui[0] - 6.*ui[1] + 4.*ui[2] - ui[3] )    
    Uj[1:Ntot-1] = 2./3.*uj[1:Ntot-1] + 1./6.*( ui[0:Ntot - 2] + ui[1:Ntot - 1] )    
    Uj[Ntot-1] = 2./3.*uj[Ntot-1] + 1./6.*( 5.*ui[Ntot-2] - 6*ui[Ntot-3] + 4.*ui[Ntot-4] - ui[Ntot-5] )
    
    Dxui=deriv1i(ui,uj[0],uj[Ntot-1],dx,ADxi)    
    Dxhj=Deriv1j(hj,dx,ADxj)
    
    rj = betaD*( ( Dxhj[0:Ntot]+Dxfj[0:Ntot] )*Dxfj[0:Ntot] + 1./2.*hj[0:Ntot]*Dxxfj[0:Ntot] )  # Auxiliary variable r at j nodes
    hi0   = 4.*hi[0]  -6.*hi [1]+4.*hi  [2]-  hi[3]                              # cubic extrapolation
    Dxui0 = 4.*Dxui[0]-6*Dxui[1]+4.*Dxui[2]-Dxui[3]                              # cubic extrapolation
    hiN   = 4.*hi  [Ntot-2]-6*hi  [Ntot-3]+4.*hi  [Ntot-4]-hi  [Ntot-5]          # cubic extrapolation
    DxuiN = 4.*Dxui[Ntot-2]-6*Dxui[Ntot-3]+4.*Dxui[Ntot-4]-Dxui[Ntot-5]          # cubic extrapolation
    
    #-
    # Cell-averaged values for auxiliary variable q at j cells
    #- 
    Qj = np.zeros( [Ntot] )
    Qj[0       ] = (1+rj[0])*Uj[0]+betaD*np.array([-1./(3.*dx*hj[0])*(hi[0]**3*Dxui[0]-hi0**3*Dxui0)-alfa/dx*fj[0]**2*(Dxui[0]-Dxui0)]) # Boundary node
    Qj[1:Ntot-1] = (1+rj[1:Ntot-1])*Uj[1:Ntot-1]+betaD*(-1./(3.*dx*hj[1:Ntot-1])*(hi[1:Ntot-1]**3*Dxui[1:Ntot-1]-hi[0:Ntot-2]**3*Dxui[0:Ntot-2])-alfa/dx*fj[1:Ntot-1]**2*(Dxui[1:Ntot-1]-Dxui[0:Ntot-2]))#Internal nodes
    Qj[Ntot-1  ] = (1+rj[Ntot-1])*Uj[Ntot-1]+betaD*(-1./(3.*dx*hj[Ntot-1])*(hiN**3*DxuiN-hi[Ntot-2]**3*Dxui[Ntot-2])-alfa/dx*fj[Ntot-1]**2*(DxuiN-Dxui[Ntot-2]))  #Boundary node
    
    #-    
    # Pentadiagonal sparse matrix to estimate pointwise cell face values
    #-
    Ajtot = VolIntMat(Ntot)

    #-
    # Cell face values q at j nodes
    #-
    qj = (spsolve( Ajtot, Qj.T ) ).T

    #-
    # Cell face values q at i nodes
    #-
    qi1   = 1./16.*( 5.*qj[0     ] + 15*qj[1     ] - 5.*qj[2     ] + 1*qj[3     ] )
    qiNm1 = 1./16.*( 5.*qj[Ntot-1] + 15*qj[Ntot-2] - 5.*qj[Ntot-3] + 1*qj[Ntot-4] )
    qi    = CellFaceInt( dx, Qj[1:Ntot - 1], ADi, qi1, qiNm1)

    #-
    # Cell averaged values for water depth
    #-

    Hi0=np.zeros(Ntot-1)
    Hi0[0:Ntot-1] = 2./3.*hi[0:Ntot-1] + 1./6.*( hj[0:Ntot-1]+hj[1:Ntot] )

    #-
    # Cell averaged values for q
    #-
    Qi0=np.zeros(Ntot-1)
    Qi0[0:Ntot-1] = 2./3.*qi[0:Ntot-1] + 1./6.*( qj[0:Ntot-1]+qj[1:Ntot] )
    
    
    # ----------------------------------------------------------------------------------------------------
    # -------------------------------- Numerical Integration ---------------------------------------------
    # --------------------- 4th order finite volume discretization for -----------------------------------
    # ------------------- spatial domain and 4th order Runge-Kutta method --------------------------------
    # ------------------------- for time integration (method of lines) -----------------------------------
    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    
    # ----------------------------------------------------------------------------------------------------
    # --------------------------------- Matrix reduction -------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    #-
#RB verifier ici 
#verifier valeur Next
    ADxjr  = Comp_1st_Mat( Next+1 )    
    ADxir  = Comp_1st_Mat( Next)    
    ADxxjr = Comp_2nd_Mat( Next+1 )    
    ADr = Comp_1st_Mat( Next-1 )    
    BQr = Cell_J_Mat( Next+1 )
    
    # ----------------------------------------------------------------------------------------------------
    # --------------------------------- Derivative estimates ---------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    #-        
    Dxhj[0:Next+1]  = Deriv1j( hj[0:Next+1], dx,ADxjr  )
    Dxxhj[0:Next+1] = Deriv2j( hj[0:Next+1], dx,ADxxjr )
    Dxuj[0:Next+1]  = Deriv1j( uj[0:Next+1], dx,ADxjr  )
    Dxxuj[0:Next+1] = Deriv2j( uj[0:Next+1], dx,ADxxjr )
    
    # ----------------------------------------------------------------------------------------------------
    # -------------------------------- Variable initialization -------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    #-
    Nu1j = np.zeros( Ntot )    
    Nu2j = np.zeros( Ntot )    
    kBrWave = np.array([])  
    t = t0    
    etamj = np.zeros(Ntot)
    

    Hi=np.zeros(Ntot-1)
    Hi[:]  = Hi0[:]

    uj0=np.zeros(Ntot)
    hj0=np.zeros(Ntot)
    hi0=np.zeros(Ntot-1)
    hj0 [:]= hj[:]
    hi0[:] = hi[:]
    uj0[:] = uj[:]
    xb = np.array([0])
    varFa = 0
    kst = 0
    hout = h0
    i = 0


    mysize=len(np.arange(t0,Tf+2*dt,dt)) 
    Time     = np.zeros( mysize )    
    hLeft    = np.zeros( mysize )    
    uLeft    = np.zeros( mysize )    
    hRight   = np.zeros( mysize )    
    uRight   = np.zeros( mysize )    
    WaterVol = np.zeros( mysize )    
    VolIn    = np.zeros( mysize ) 

    Qi= np.zeros(Ntot-1) 

#    print Tf
#    return
    while t <= Tf +dt:

        # ------------------------------------------------------------------------------------------------
        # --------------- Storing time, boundary conditions and total water volume in the domain ---------
        # ------------------------------------------------------------------------------------------------
        #-
        Time[i] = t
        hLeft[i]  = hj[0]
        uLeft[i]  = uj[0]
        hRight[i] = hj[N-1]
        uRight[i] = uj[N-1]
        WaterVol[i] = dx*np.sum( Hi[0:N-1] )
        VolIn[i] = dt*hLeft[i]*uLeft[i]
        i = i + 1
        
        # ------------------------------------------------------------------------------------------------
        # ------------------------------- Storing computation data ---------------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        vv = kst / STO
        if ( vv == 1 or t == 0 ):

            set_interp = interp1d( Xj[Next-10:Next+1], hj[Next-10:Next+1], kind='linear', fill_value='extrapolate' )
            Hw = np.concatenate( [ hj[0:Next+1], set_interp( Xj[Next+1:Ntot] ) ] ) 
            
            set_interp = interp1d( Xi[Next-11:Next],  hi[Next-11:Next], kind='linear', fill_value='extrapolate' )
            Hwi = np.concatenate( [ hj[0:Next], set_interp( Xi[Next:Ntot-1] ) ] ) 
            
            set_interp = interp1d( Xj[Next-10:Next+1], uj[Next-10:Next+1], kind='linear', fill_value='extrapolate' )
            Uw = np.concatenate( [ uj[0:Next+1], set_interp( Xj[Next+1:Ntot] ) ] ) 
            
            Uxw  = np.concatenate( [ Dxuj [0:Next+1], np.zeros(Ntot - Next-1) ] )
            Uxxw = np.concatenate( [ Dxxuj[0:Next+1], np.zeros(Ntot - Next-1) ] )
            NU1w = np.concatenate( [ Nu2j[0:Next+1] , np.zeros(Ntot - Next-1) ] )
                        
            
   #         for idx in range(len(hj[0:Next+1])):
   #             fidHj.write  ( '%6.8f \t' % hj[idx])
   #         fidHj.write  ('\n')
   #         for idx in range(len(uj[0:Next+1])):
   #             fidUj.write  ( '%6.8f \t' % uj[idx])
   #         fidUj.write  ('\n')


            for idx in range(len(Hw)):
                fidHj.write  ( '%6.8f \t' % Hw[idx])
            fidHj.write  ('\n')
            
            if full_output:
                for idx in range(len(Uw)):
                    fidUj.write  ( '%6.8f \t' % Uw[idx])
                fidUj.write  ('\n')

                for idx in range(len(Hwi)):
                    fidHi.write  ( '%6.8f \t' % Hwi [idx])
                fidHi.write  ('\n')

        #    for idx in range(len(uj[0:Next+1])):
        #        fidUj.write  ( '%6.8f \t' % uj[idx])
        #    fidUj.write  ('\n')

                for idx in range(len(Uxw)):
                    fidUxj.write ( '%6.8f \t' % Uxw [idx])
                fidUxj.write ('\n')

                for idx in range(len(Uxxw)):
                    fidUxxj.write( '%6.8f \t' % Uxxw[idx])
                fidUxxj.write('\n')

            for idx in range(len(NU1w)):
               fidNUj.write ( '%6.8f \t' % NU1w [idx])
            fidNUj.write ('\n')

            fidTime.write( '%6.4f \t' % t    )
            fidTime.write('\n')
            toto=N+1
            fidNwet.write( '%6.4f \t' % toto )
            fidNwet.write('\n')

            fidXb.write  ( '%6.4f \t' % xb   )
            fidXb.write  ('\n')

            fidHout.write( '%6.4f \t' % hout )
            fidHout.write('\n')

            kst = -1
        kst = kst + 1
        # ------------------------------------------------------------------------------------------------
        # --------------------------------- Screen Information -------------------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
#RB    
        os.system('clear') 
        print('%s %6.3f %s \t'   % ('Current time = ',t,'s'))
        #print 'i =', i
        print('%s %6.3f %s \t'   % ('dx = ',dx,'m'))
        print('%s %6.3f %s \t'   % ('dt = ',dt,'s'))
        print('%s %6.3f %s \t'   % ('Cr = ',Cr,'m/s'))
        print('%s %6.0f \n\n'    % ('N  = ',N))
        print('%s %6.0f \n\n'    % ('Iter  = ',miter))
        print('%s %6.3f %s \n'   % ('Breaking point x-coordinate xb = ',xb,'m'))
        print( '%s %6.3f %s \n'  % ('Volume of water in the domain V/V0         = ', 100*WaterVol[i-1]/WaterVol[0], '%' ))
        print('%s %6.3f %s \n\n' % ('Volume of water entering the domain Vin/V0 = ', 100*VolIn[i-1]/WaterVol[0]   , '%' ))

        # ------------------------------------------------------------------------------------------------
        # ------------------------------ First Runge-Kutta stage -----------------------------------------
        # ------------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------
        # ----------------------- Indentify wave crest and trough positions ------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        MatWav=np.array([])
        jc,jt,MatWav = WaveProp( Xj[0:N+1], hj[0:N+1], fj[0:N+1], Dxhj[0:N+1], Dxfj[0:N+1], WDtol )


        # ------------------------------------------------------------------------------------------------
        # ---------------------- Actualization of the breaking event index -------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        kBrWave = BrWaveIndex( t, kTb, PHIb, PHIf, hj[0:N+1], uj[0:N+1], fj[0:N+1], MatWav, kBrWave, dx, dt, g, WDtol )

        # ------------------------------------------------------------------------------------------------
        # ------------------- Breaking criterium and eddy viscosity coefficients -------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        Nu1j[0:N+1], Nu2j[0:N+1], xb = EddyViscosity( t, PHI1, PHI2, bCointe, alfab, alfaf, gamab, gamaf, kap1, kap2, kBrWave, Xj[0:N+1], hj[0:N+1], fj[0:N+1], dx, g, WDtol )
        Nu1j[:] = np.concatenate( [ Nu1j[0:N], np.zeros(Ntot - N ) ] )
        Nu2j[:] = np.concatenate( [ Nu2j[0:N], np.zeros(Ntot - N ) ] )

        # ------------------------------------------------------------------------------------------------
        # -------------------------- Flux function for continuity equation  ------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        FCj[0:N+1] = FluxContinuity( hj[0:N+1], uj[0:N+1], Dxhj[0:N+1], Dxfj[0:N+1], Nu1j[0:N+1] )
        FCi[0:N] = 1./dx*( FCj[1:N+1] - FCj[0:N] )
        FCj[N+1::] = 0.
        FCi[N::] = 0.

        # ------------------------------------------------------------------------------------------------
        # ---------------------------- Flux function for momemtum equation  ------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        FMj[0:N+1] = FluxMomentum( qj[0:N+1], hj[0:N+1], fj[0:N+1], uj[0:N+1], Dxhj[0:N+1], Dxfj[0:N+1], Dxuj[0:N+1], Dxxhj[0:N+1], Dxxuj[0:N+1], Dxxfj[0:N+1], g, betaD, alfa, WDtol )
        FMi[0:N] = 1./dx*( FMj[1:N+1] - FMj[0:N] )
        FMj[N+1::] = 0.
        FMi[N::]=  0.

        # ------------------------------------------------------------------------------------------------
        # --------------------------- Source function for momemtum equation ------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        SMi[0:N] = SourceMomentum( hi[0:N], hj[0:N+1], fi[0:N], fj[0:N+1], uj[0:N+1], Dxhj[0:N+1], Dxfj[0:N+1], Dxfi[0:N], Dxuj[0:N+1], Nu2j[0:N+1], Xi[0:N], g, betaD, alfa, fricFact, Cfi[0:N], WDtol )
        SMi = 1./dx*SMi
        DtHi = -FCi
        DtQi = -FMi + SMi

        # ------------------------------------------------------------------------------------------------
        # ------------------------------- Estimates of Hi and Qi at t=t+dt/2 -----------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        kH1 = dt*DtHi
        kQ1 = dt*DtQi
        Hi[0:N] = Hi0[0:N] + 1./2.*kH1[0:N]
        Qi[0:N] = Qi0[0:N] + 1./2.*kQ1[0:N]
        if FilterD1 == 1:
            Hi[N-NIZ:N] = explicitFILT0( Hi[N-NIZ:N] )
            Qi[N-NIZ:N] = explicitFILT0( Qi[N-NIZ:N] )
        
        #-            
        # Linear extrapolation for dry cells
        #-
        dh = 1./dx*( Hi[N-1] - Hi[N-2] )
        dq = 1./dx*( Qi[N-1] - Qi[N-2] )
        Hi[N:Ntot] = Hi[N-1] + dh*( Xi[N:Ntot] - Xi[N-1] )
        Qi[N:Ntot] = Qi[N-1] + dq*( Xi[N:Ntot] - Xi[N-1] )

        # ------------------------------------------------------------------------------------------------
        # ---------------------- Cell faces values hj, qj and uj at t=t+dt/2 -----------------------------
        # ------------------------------------------------------------------------------------------------
        
        # ------------------------------------------------------------------------------------------------
        # Left boundary condition for h and u at t=t+dt/2 (explicit stage)
        # ------------------------------------------------------------------------------------------------
        #-
        
        #-
        # Initial location of left going characteristic (xR=beta*dx) + water and depth-averaged velocity values
        #-

#RB
#RB Verif

        data=(dx, dt/2., uj0, hj0, g)
        beta = fsolve ( findposRL1, Cr, args=data )


        p = int(floor(beta))
        b = beta - p
        uR0 = (1-b)*uj0[p] + b*uj0[p+1];
        hR0 = (1-b)*hj0[p] + b*hj0[p+1];    
        DxfR0 = (1-b)*Dxfj[p] + b*Dxfj[p+1];
     
        #Water depth and velocity for incident wave
        hL0 = DataManip( t+dt/2., h0+offset, DAT )
        uL0 = (g*hL0)**0.5*(hL0+fj[0])/hL0
        
        IR = uR0 -2*(g*hR0)**0.5 - dt/2.*g*DxfR0 # Variable de Riemann associee a C-

        #-
        # Riemann variable for right going characteristic
        #-
        IL = uL0 + 2.*(g*hL0) ** 0.5   #- dot(dot(dot(0,dt)/2.,g),Dxfj(1))

        #-
        # Boundary values for water depth and depth-averaged velocity at t=t+dt/2
        #-        
        uA = 1./2.*(IL + IR)
        hA = 1./(16.*g)*(IL - IR)**2
        
        #-
        # Cell face values reconstruction for water depth h
        #-
        hj[0:Next+1] = CellFaceInt( dx, Hi[0:Next], ADr, hA )
        Dxhj[0:Next+1]  = Deriv1j( hj[0:Next+1], dx, ADxjr  )
        Dxxhj[0:Next+1] = Deriv2j( hj[0:Next+1], dx, ADxxjr )
        hj[Next+1::]=0
        Dxhj[Next+1::]=0
        Dxxhj[Next+1::]=0

        #-
        # Water depth i-values
        #-

        hi[0:Next] = 3./2.*Hi[0:Next] - 1./4.*( hj[0:Next] + hj[1:Next+1] )
        hi[Next::] = 0.
        uj[0:Next+1], qj[0:Next+1] = DeffCorrU( uA, Qi[0:Next], hi[0:Next], hj[0:Next+1], fj[0:Next+1], Dxhj[0:Next+1], Dxfj[0:Next+1], Dxxfj[0:Next+1], dx, ADr, BQr, uj[0:Next+1], betaD, alfa, tol, MaxIter, WDtol )        
        uj[Next+1::]=0
        qj[Next+1::]=0
        Dxuj [0:Next+1] = Deriv1j( uj[0:Next+1], dx, ADxjr  )
        Dxxuj[0:Next+1] = Deriv2j( uj[0:Next+1], dx, ADxxjr )
     #    print 'len uj 0', len(uj[0:Next+1])
        Dxuj[Next+1::]=0
        Dxxuj[Next+1::]=0


        # ------------------------------------------------------------------------------------------------
        # ----------------------------- Second Runge-Kutta stage -----------------------------------------
        # ------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------
    

        # ------------------------------------------------------------------------------------------------
        # ----------------------- Indentify wave crest and trough positions ------------------------------
        # ------------------------------------------------------------------------------------------------
        #-

        MatWav=np.array([])
        jc, jt, MatWav = WaveProp( Xj[0:N+1], hj[0:N+1], fj[0:N+1], Dxhj[0:N+1], Dxfj[0:N+1], WDtol )

        # ------------------------------------------------------------------------------------------------
        # ---------------------- Actualization of the breaking event index -------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        kBrWave = BrWaveIndex( t, kTb, PHIb, PHIf, hj[0:N+1], uj[0:N+1], fj[0:N+1], MatWav, kBrWave, dx, dt, g, WDtol )

 

        # ------------------------------------------------------------------------------------------------
        # ------------------- Breaking criterium and eddy viscosity coefficients -------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        Nu1j[0:N+1], Nu2j[0:N+1], xb = EddyViscosity( t, PHI1, PHI2, bCointe, alfab, alfaf, gamab, gamaf, kap1, kap2, kBrWave, Xj[0:N+1], hj[0:N+1], fj[0:N+1], dx, g, WDtol )
        Nu1j[:] = np.concatenate( [ Nu1j[0:N], np.zeros(Ntot - N ) ] )
        Nu2j[:] = np.concatenate( [ Nu2j[0:N], np.zeros(Ntot - N ) ] )




        # ------------------------------------------------------------------------------------------------
        # -------------------------- Flux function for continuity equation  ------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        FCj[0:N+1] = FluxContinuity( hj[0:N+1], uj[0:N+1], Dxhj[0:N+1], Dxfj[0:N+1], Nu1j[0:N+1] )
        FCi[0:N] = 1./dx*( FCj[1:N+1] - FCj[0:N] )
        FCj[N+1::] = 0
        FCi[N::] = 0.

        # ------------------------------------------------------------------------------------------------
        # ---------------------------- Flux function for momemtum equation  ------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        FMj[0:N+1] = FluxMomentum( qj[0:N+1], hj[0:N+1], fj[0:N+1], uj[0:N+1], Dxhj[0:N+1], Dxfj[0:N+1], Dxuj[0:N+1], Dxxhj[0:N+1], Dxxuj[0:N+1], Dxxfj[0:N+1], g, betaD, alfa, WDtol )
        FMj[N+1::] = 0.
        FMi[0:N] = 1./dx*( FMj[1:N+1] - FMj[0:N] )
        FMi[N::] = 0.
        # ------------------------------------------------------------------------------------------------
        # --------------------------- Source function for momemtum equation ------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        # precision ici
        SMi[0:N] = SourceMomentum( hi[0:N], hj[0:N+1], fi[0:N], fj[0:N+1], uj[0:N+1], Dxhj[0:N+1], Dxfj[0:N+1], Dxfi[0:N], Dxuj[0:N+1], Nu2j[0:N+1], Xi[0:N], g, betaD, alfa, fricFact, Cfi[0:N], WDtol )
        SMi[N::] = 0.
        SMi = 1./dx*SMi
        DtHi = -FCi
        DtQi = - FMi + SMi


        # ------------------------------------------------------------------------------------------------
        # ------------------------------- Estimates of Hi and Qi at t=t+dt/2 -----------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        kH2 = dt*DtHi
        kQ2 = dt*DtQi
        Hi[0:N] = Hi0[0:N] + 1./2.*kH2[0:N]
        Qi[0:N] = Qi0[0:N] + 1./2.*kQ2[0:N]
        if FilterD1 == 1:
            Hi[N-NIZ:N] = explicitFILT0( Hi[N-NIZ:N] )
            Qi[N-NIZ:N] = explicitFILT0( Qi[N-NIZ:N] )
        
        #-            
        # Linear extrapolation for dry cells
        #-
        dh = 1./dx*(Hi[N-1] - Hi[N-2])
        dq = 1./dx*(Qi[N-1] - Qi[N-2])
        Hi[N:Ntot] = Hi[N-1] + dh*( Xi[N:Ntot] - Xi[N-1] )
        Qi[N:Ntot] = Qi[N-1] + dq*( Xi[N:Ntot] - Xi[N-1] )


        # ------------------------------------------------------------------------------------------------
        # ----------------------------------- Finding last wet cell --------------------------------------
        # ------------------------------------------------------------------------------------------------
        #-        
        N1 = np.max(np.where(hj > WDtol))
        N2 = np.min(np.where(hj < WDtol))
        Nm = min(N1,N2)
#RB
        if ( Nm.size > 0) and Nm != N and Nm < (Ntot-1 - NSZ):
            if Nm > N:
                set_interp = interp1d( Xi[0:N], kH1[0:N], kind='linear', fill_value='extrapolate' )
                toto = set_interp(Xi[N:Nm])
                kH1 = np.concatenate([kH1[0:N],toto ])

                set_interp = interp1d( Xi[0:N], kH2[0:N], kind='linear', fill_value='extrapolate' )
                toto = set_interp(Xi[N:Nm])
                kH2 = np.concatenate([kH2[0:N],toto ])

                set_interp = interp1d( Xi[0:N], kQ1[0:N], kind='linear', fill_value='extrapolate' )
                toto = set_interp(Xi[N:Nm])
                kQ1 = np.concatenate([kQ1[0:N],toto ])

                set_interp = interp1d( Xi[0:N], kQ2[0:N], kind='linear', fill_value='extrapolate' )
                toto = set_interp(Xi[N:Nm])
                kQ2 = np.concatenate([kQ2[0:N],toto ])

      #          set_interp = interp1d( Xj[0:N+1], etamj[0:N+1], kind='linear', fill_value='extrapolate' )
      #          toto = set_interp(Xj[N+1:Nm+1])                
      #          etamj = np.concatenate([etamj[0:N],toto ])
            N = Nm
            Next = N + NSZ
            Nint = N - NSZ


            #-
            # Matrix reduction
            #-
            ADxjr  = Comp_1st_Mat( Next+1 )    
            ADxir  = Comp_1st_Mat( Next)    
            ADxxjr = Comp_2nd_Mat( Next+1 )    
            ADr = Comp_1st_Mat( Next-1 )    
            BQr = Cell_J_Mat( Next+1 )

 
        # ------------------------------------------------------------------------------------------------
        # ---------------------- Cell faces values hj, qj and uj at t=t+dt/2 -----------------------------
        # ------------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------
        # Left boundary condition for h and u at t=t+dt/2 (implicit stage)
        # ------------------------------------------------------------------------------------------------
        #-

        #-
        # Initial location of left going characteristic (xR=beta*dx) + water and depth-averaged velocity values
        #-
                
        data = ( dx, dt/2., uj0, hj0, hj[0], uj[0],g )
 #   beta=fzero('findposRL2',[0 Cr],options,dx,dt/2,uj0,hj0,hj(1),uj(1),g);
        beta = fsolve ( findposRL2, Cr, args=data ) 
        p = int(floor(beta))
        b = beta - p
        uR0 = (1-b)*uj0[p] + b*uj0[p+1];
        hR0 = (1-b)*hj0[p] + b*hj0[p+1];    
        DxfR0 = (1-b)*Dxfj[p] + b*Dxfj[p+1];
     
        #Water depth and velocity for incident wave
        hL0 = DataManip( t+dt/2., h0+offset,DAT )
        uL0 = (g*hL0)**0.5*(hL0+fj[0])/hL0
   
        IR = uR0 -2*(g*hR0)**0.5 - dt/2.*g*DxfR0 # Variable de Riemann associee a C-
     
        #-
        # Riemann variable for right going characteristic
        #-
        IL = uL0 + 2.*(g*hL0) ** 0.5   #- dot(dot(dot(0,dt)/2.,g),Dxfj(1))
        
        #-
        # Boundary values for water depth and depth-averaged velocity at t=t+dt/2
        #-
        uA = 1./2.*(IL + IR)
        hA = 1./(16.*g)*(IL - IR)**2

        # ------------------------------------------------------------------------------------------------
        # Cell face values reconstruction for water depth h
        hj[0:Next+1] = CellFaceInt( dx, Hi[0:Next], ADr, hA )
        Dxhj [0:Next+1] = Deriv1j( hj[0:Next+1], dx, ADxjr  )
        Dxxhj[0:Next+1] = Deriv2j( hj[0:Next+1], dx, ADxxjr )
        hj   [Next+1::] = 0
        Dxhj [Next+1::] = 0
        Dxxhj[Next+1::] = 0

        hi[0:Next] = 3./2.*Hi[0:Next] - 1./4.*(hj[0:Next] + hj[1:Next+1])
        hi[Next::] = 0.

        if Next > len(uj)-1:
            Nu = len(uj)-1
            du = 1./dx*( uj[Nu]-uj[Nu-1] )
            uj[Nu+1:Next+1] = du*( Xj[Nu+1:Next+1] - Xj[Next] )
            if FilterD1 == 1:
                uj[Nu-NIZ:Next+1] = explicitFILT0( uj[Nu-NIZ,Next+1] )
        
        uj[0:Next+1], qj[0:Next+1] = DeffCorrU( uA, Qi[0:Next], hi[0:Next], hj[0:Next+1], fj[0:Next+1], Dxhj[0:Next+1], Dxfj[0:Next+1], Dxxfj[0:Next+1], dx, ADr, BQr, uj[0:Next+1], betaD, alfa, tol, MaxIter, WDtol );
        Dxuj [0:Next+1] = Deriv1j( uj[0:Next+1], dx, ADxjr  )
        Dxxuj[0:Next+1] = Deriv2j( uj[0:Next+1], dx, ADxxjr )
        uj[Next+1::]=0
        qj[Next+1::]=0
        Dxuj[Next+1::]=0
        Dxxuj[Next+1::]=0
       
        hj0[:] = hj[:]
        hi0[:] = hi[:]
        uj0[:] = uj[:]
 

        # ------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------
        # ------------------------------ Third Runge-Kutta stage -----------------------------------------
    
        # ------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------
        # ----------------------- Indentify wave crest and trough positions ------------------------------
        #-
        MatWav=np.array([])
        jc,jt,MatWav = WaveProp( Xj[0:N+1], hj[0:N+1],fj[0:N+1], Dxhj[0:N+1], Dxfj[0:N+1], WDtol )

        # ------------------------------------------------------------------------------------------------
        # ---------------------- Actualization of the breaking event index -------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        kBrWave = BrWaveIndex( t, kTb, PHIb, PHIf, hj[0:N+1], uj[0:N+1], fj[0:N+1], MatWav, kBrWave, dx, dt, g, WDtol )
        
        # ------------------------------------------------------------------------------------------------
        # ------------------- Breaking criterium and eddy viscosity coefficients -------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        Nu1j[0:N+1], Nu2j[0:N+1], xb = EddyViscosity(t, PHI1, PHI2, bCointe, alfab, alfaf, gamab, gamaf, kap1, kap2, kBrWave, Xj[0:N+1], hj[0:N+1],fj[0:N+1], dx, g, WDtol )
        Nu1j[:] = np.concatenate( [ Nu1j[0:N], np.zeros(Ntot - N ) ] )
        Nu2j[:] = np.concatenate( [ Nu2j[0:N], np.zeros(Ntot - N ) ] )
        
        # ------------------------------------------------------------------------------------------------
        # -------------------------- Flux function for continuity equation  ------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        FCj[0:N+1] = FluxContinuity( hj[0:N+1], uj[0:N+1], Dxhj[0:N+1], Dxfj[0:N+1], Nu1j[0:N+1] )
        FCj[N+1::] = 0.
        FCi[0:N] = 1./dx*( FCj[1:N+1] - FCj[0:N] )
        FCi[N::] = 0.

        # ------------------------------------------------------------------------------------------------
        # ---------------------------- Flux function for momemtum equation  ------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        FMj[0:N+1]=FluxMomentum(qj[0:N+1], hj[0:N+1],fj[0:N+1], uj[0:N+1], Dxhj[0:N+1], Dxfj[0:N+1],Dxuj[0:N+1], Dxxhj[0:N+1], Dxxuj[0:N+1], Dxxfj[0:N+1], g, betaD, alfa, WDtol )
        FMj[N+1::] = 0.
        FMi[0:N] = 1./dx*( FMj[1:N+1] - FMj[0:N] )
        FMi[N::] = 0.

        # ------------------------------------------------------------------------------------------------
        # --------------------------- Source function for momemtum equation ------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        SMi[0:N]=SourceMomentum( hi[0:N], hj[0:N+1], fi[0:N], fj[0:N+1], uj[0:N+1], Dxhj[0:N+1], Dxfj[0:N+1], Dxfi[0:N], Dxuj[0:N+1], Nu2j[0:N+1], Xi[0:N], g, betaD, alfa, fricFact, Cfi[0:N], WDtol )
        SMi[N::] = 0.
        SMi = 1./dx*SMi
    
        DtHi = - FCi
        DtQi = -FMi + SMi

        # ------------------------------------------------------------------------------------------------
        # --------------------------------- Estimates of Hi and Qi at t=t+dt -----------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        kH3 = dt*DtHi
        kQ3 = dt*DtQi
        Hi[0:N] = Hi0[0:N] + kH3[0:N]
        Qi[0:N] = Qi0[0:N] + kQ3[0:N]
        if FilterD1 == 1:
            Hi[N-NIZ:N] = explicitFILT0( Hi[N-NIZ:N] )
            Qi[N-NIZ:N] = explicitFILT0( Qi[N-NIZ:N] )
        
        #-            
        # Linear extrapolation for dry cells
        #-
        dh = 1./dx*(Hi[N-1] - Hi[N-2])
        dq = 1./dx*(Qi[N-1] - Qi[N-2])
        Hi[N:Ntot] = Hi[N-1] + dh*(Xi[N:Ntot] - Xi[N-1])
        Qi[N:Ntot] = Qi[N-1] + dq*(Xi[N:Ntot] - Xi[N-1])
       

        # ------------------------------------------------------------------------------------------------
        # ------------------------ Cell faces values hj, qj and uj at t=t+dt -----------------------------
        # ------------------------------------------------------------------------------------------------
        
        # ------------------------------------------------------------------------------------------------
        # Left boundary condition for h and u at t=t+dt (explicit stage)
        # ------------------------------------------------------------------------------------------------
        #
        
        #-
        # Initial location of left going characteristic (xR=beta*dx) + water and depth-averaged velocity values
        #-
        #beta=fzero('findposRL1',[0 Cr],options,dx,dt/2,uj0,hj0,g);
        data=(dx,dt/2.,uj0,hj0,g)
        beta = fsolve ( findposRL1, 0,  args=data ) 
        
        p = int(floor(beta))
        b = beta - p
        uR0 = (1-b)*uj0[p] + b*uj0[p+1];
        hR0 = (1-b)*hj0[p] + b*hj0[p+1];    
        DxfR0 = (1-b)*Dxfj[p] + b*Dxfj[p+1];
     
        #Water depth and velocity for incident wave
        hL0 = DataManip( t+dt, h0+offset,DAT )
        uL0 = (g*hL0)**0.5*(hL0+fj[0])/hL0
   
        IR = uR0 -2*(g*hR0)**0.5 - dt/2.*g*DxfR0 # Variable de Riemann associee a C-
     
        #-
        # Riemann variable for right going characteristic
        #-
        IL = uL0 + 2.*(g*hL0) ** 0.5   #- dot(dot(dot(0,dt)/2.,g),Dxfj(1))
        
        #-
        # Boundary values for water depth and depth-averaged velocity at t=t+dt/2
        #-
        uA = 1./2.*(IL + IR)
        hA = 1./(16.*g)*(IL - IR)**2


        # Cell face values reconstruction for water depth h
        hj[0:Next+1] = CellFaceInt( dx, Hi[0:Next], ADr, hA )
     #   print 'len hj third RK', len(hj[0:Next+1])
        Dxhj [0:Next+1] = Deriv1j( hj[0:Next+1], dx, ADxjr  )
        Dxxhj[0:Next+1] = Deriv2j( hj[0:Next+1], dx, ADxxjr )
        hj   [Next+1::] = 0.
        Dxhj [Next+1::] = 0.
        Dxxhj[Next+1::] = 0.

        hi[0:Next] = 3./2.*Hi[0:Next] - 1./4.*(hj[0:Next] + hj[1:Next+1])
        hi[Next::] = 0.


        uj[0:Next+1], qj[0:Next+1] = DeffCorrU(uA, Qi[0:Next], hi[0:Next], hj[0:Next+1], fj[0:Next+1], Dxhj[0:Next+1], Dxfj[0:Next+1], Dxxfj[0:Next+1], dx, ADr, BQr, uj[0:Next+1], betaD, alfa, tol, MaxIter, WDtol);
        Dxuj[0:Next+1] = Deriv1j( uj[0:Next+1], dx, ADxjr  )
        Dxxuj[0:Next+1]= Deriv2j( uj[0:Next+1], dx, ADxxjr )
        uj  [Next+1::] = 0
        qj  [Next+1::] = 0
        Dxuj[Next+1::] = 0
        Dxxuj[Next+1::] = 0




        # ------------------------------------------------------------------------------------------------
        # ----------------------------- Fourth Runge-Kutta stage -----------------------------------------
        # ------------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------
        # ----------------------- Indentify wave crest and trough positions ------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        MatWav=np.array([])
        jc,jt,MatWav = WaveProp( Xj[0:N+1], hj[0:N+1], fj[0:N+1], Dxhj[0:N+1], Dxfj[0:N+1], WDtol )
        # ------------------------------------------------------------------------------------------------
        # ---------------------- Actualization of the breaking event index -------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        kBrWave = BrWaveIndex( t, kTb, PHIb, PHIf, hj[0:N+1], uj[0:N+1], fj[0:N+1], MatWav, kBrWave, dx, dt, g, WDtol )

        # ------------------------------------------------------------------------------------------------
        # ------------------- Breaking criterium and eddy viscosity coefficients -------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        Nu1j[0:N+1],Nu2j[0:N+1],xb = EddyViscosity( t, PHI1, PHI2, bCointe, alfab, alfaf, gamab, gamaf, kap1, kap2, kBrWave, Xj[0:N+1], hj[0:N+1], fj[0:N+1], dx, g, WDtol )
        Nu1j[:] = np.concatenate( [ Nu1j[0:N], np.zeros(Ntot - N ) ] )
        Nu2j[:] = np.concatenate( [ Nu2j[0:N], np.zeros(Ntot - N ) ] )

        # ------------------------------------------------------------------------------------------------
        # -------------------------- Flux function for continuity equation  ------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        FCj[0:N+1] = FluxContinuity( hj[0:N+1], uj[0:N+1], Dxhj[0:N+1], Dxfj[0:N+1], Nu1j[0:N+1] )
        FCi[0:N] = 1./dx*( FCj[1:N+1] - FCj[0:N] )
        FCj[N+1::]=0 
        FCj[N::]=0 

        # ------------------------------------------------------------------------------------------------
        # ---------------------------- Flux function for momemtum equation  ------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        FMj[0:N+1] = FluxMomentum( qj[0:N+1], hj[0:N+1], fj[0:N+1], uj[0:N+1], Dxhj[0:N+1], Dxfj[0:N+1], Dxuj[0:N+1], Dxxhj[0:N+1], Dxxuj[0:N+1], Dxxfj[0:N+1], g, betaD, alfa, WDtol )
        FMi[0:N] = 1./dx*( FMj[1:N+1] - FMj[0:N] )
        FMj[N+1::]=0 
        FMj[N::]=0 
        # ------------------------------------------------------------------------------------------------
        # --------------------------- Source function for momemtum equation ------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        SMi[0:N] = SourceMomentum( hi[0:N], hj[0:N+1], fi[0:N], fj[0:N+1], uj[0:N+1], Dxhj[0:N+1], Dxfj[0:N+1], Dxfi[0:N], Dxuj[0:N+1], Nu2j[0:N+1], Xi[0:N], g, betaD, alfa, fricFact, Cfi[0:N], WDtol )
        SMi[N::]=0 
        SMi = 1./dx*SMi
        DtHi = -FCi
        DtQi = -FMi + SMi

        # ------------------------------------------------------------------------------------------------
        # ------------------------------- Estimates of Hi and Qi at t=t+dt -------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        kH4 = dt*DtHi
        kQ4 = dt*DtQi
        Hi[0:N] = Hi0[0:N] + 1./2.*kH4[0:N]
        Qi[0:N] = Qi0[0:N] + 1./2.*kQ4[0:N]
        Hi[0:N]=Hi0[0:N]+(kH1[0:N]+2*kH2[0:N]+2*kH3[0:N]+kH4[0:N])/6.
        Qi[0:N]=Qi0[0:N]+(kQ1[0:N]+2*kQ2[0:N]+2*kQ3[0:N]+kQ4[0:N])/6.

        if FilterD1 == 1:
            Hi[N-NIZ:N] = explicitFILT0( Hi[N-NIZ:N] )
            Qi[N-NIZ:N] = explicitFILT0( Qi[N-NIZ:N] )
        
        #-            
        # Linear extrapolation for dry cells
        #-
        dh = 1./dx*(Hi[N-1] - Hi[N-2])
        dq = 1./dx*(Qi[N-1] - Qi[N-2])
        Hi[N:Ntot] = Hi[N-1] + dh*(Xi[N:Ntot] - Xi[N-1])
        Qi[N:Ntot] = Qi[N-1] + dq*(Xi[N:Ntot] - Xi[N-1])
        
        #-
        # High order linear filter to dump out small peturbations
        #-
        if FilterD2 == 1:
            if NFT == 0:
                dum = 0
                varFn = varFa
            else:
                dum = 1
                varFn = int(floor( t / (NFT*dt) ))
            if ((varFn > varFa or dum == 0)):
                Hi = implicitFILT8( Hi, alfaF )
                Qi = implicitFILT8( Qi, alfaF )
            varFa = varFn

        Hi0[:] = Hi[:]
        Qi0[:] = Qi[:]

                # ------------------------------------------------------------------------------------------------
        # ----------------------------------- Finding last wet cell --------------------------------------
        # ------------------------------------------------------------------------------------------------
        #-
        N1 = np.max( np.where(hj > WDtol) )
        N2 = np.min( np.where(hj < WDtol) ) 
        Nm = min(N1,N2)
        if ( Nm.size > 0 and Nm != N and Nm < Ntot - NSZ ):        
        #    if Nm > N:
        #        print np.shape(Xj[0:N+1])
        #        print np.shape(etamj[0:N+1])
        #        set_interp = interp1d( Xj[0:N+1], etamj[0:N+1], kind='linear', fill_value='extrapolate' )
        #        toto = set_interp( Xj[N+1:Nm+1] )
        #        etamj = np.concatenate( [etamj[0:N],toto ] )
            N = Nm
            Next = N + NSZ 
            Nint = N - NSZ  
            #-
            # Matrix reduction
            #-
            ADxjr  = Comp_1st_Mat( Next+1 )
            ADxir  = Comp_1st_Mat( Next)
            ADxxjr = Comp_2nd_Mat( Next+1 )
            ADr = Comp_1st_Mat( Next-1 )
            BQr = Cell_J_Mat( Next+1 )        
  #      if i== 482:
  #          print Nm
  #          print Next
  #          return

        # ------------------------------------------------------------------------------------------------
        # ------------------------ Cell faces values hj, qj and uj at t=t+dt -----------------------------
        # ------------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------------------
        # Left boundary condition for h and u at t=t+dt (implicit stage)
        # ------------------------------------------------------------------------------------------------
        #-

        #-
        # Initial location of left going characteristic (xR=beta*dx) + water and depth-averaged velocity values
        #-
        data=( dx, dt/2., uj0, hj0, hj[0], uj[0], g )
        beta = fsolve ( findposRL2, Cr, args=data )

        p = int(floor(beta))
        b = beta - p
        uR0 = (1-b)*uj0[p] + b*uj0[p+1];
        hR0 = (1-b)*hj0[p] + b*hj0[p+1];
        DxfR0 = (1-b)*Dxfj[p] + b*Dxfj[p+1];

        #Water depth and velocity for incident wave
        hL0 = DataManip( t+dt, h0+offset,DAT )
        uL0 = (g*hL0)**0.5*(hL0+fj[0])/hL0

        IR = uR0 -2*(g*hR0)**0.5 - dt/2.*g*DxfR0 # Variable de Riemann associee a C-        
        #-
        # Riemann variable for right going characteristic
        #-
        IL = uL0 + 2.*(g*hL0)** 0.5   #- dot(dot(dot(0,dt)/2.,g),Dxfj(1))

        #-
        # Boundary values for water depth and depth-averaged velocity at t=t+dt/2
        #-
        uA = 1./2.*(IL + IR)
        hA = 1./(16.*g)*(IL - IR)**2

        hout = 1./(16.*g)*(2*(g*h0)**0.5-dt/2.*g*Dxfj[0]-IR)**2

        #-
        # Cell face values reconstruction for water depth h
        #-
        hj[0:Next+1] = CellFaceInt( dx, Hi[0:Next], ADr, hA )
        hj[Next+1::]=0
        Dxhj [0:Next+1] = Deriv1j( hj[0:Next+1], dx, ADxjr  )
        Dxxhj[0:Next+1] = Deriv2j( hj[0:Next+1], dx, ADxxjr )
        Dxhj[Next+1::]=0
        Dxhj[Next+1::]=0

        hi[0:Next] = 3./2.*Hi[0:Next] - 1./4.*(hj[0:Next] + hj[1:Next+1])
        hi[Next::]=0

   #     print 'Next =', Next
   #     print 'len hj',  len(hj[0:Next+1] ), len(hj), aa
   #     print 'len uj 1',  len(uj[0:Next+1] ), len(uj), bb
        if Next > len(uj)-1:
            Nu = len(uj)-1
            du = 1./dx*( uj[Nu]-uj[Nu-1] )
            uj[Nu+1:Next+1] = du*( Xj[Nu+1:Next+1] - Xj[Next] )
            if FilterD1 == 1:
                uj[Nu-NIZ:Next+1] = explicitFILT0( uj[Nu -NIZ:Next+1] )

   #     print 'len uj 2',  len(uj[0:Next+1] )

        uj[0:Next+1], qj[0:Next+1] = DeffCorrU(uA, Qi[0:Next], hi[0:Next], hj[0:Next+1], fj[0:Next+1], Dxhj[0:Next+1], Dxfj[0:Next+1], Dxxfj[0:Next+1], dx, ADr, BQr, uj[0:Next+1], betaD, alfa, tol, MaxIter, WDtol)
        Dxuj [0:Next+1] = Deriv1j( uj[0:Next+1], dx, ADxjr  )
        Dxxuj[0:Next+1] = Deriv2j( uj[0:Next+1], dx, ADxxjr )
#        print uj[0:30]
        uj[Next+1::]=0
        qj[Next+1::]=0
        Dxuj[Next+1::]=0
        Dxuj[Next+1::]=0

        hj0[:] = hj[:]
        hi0[:] = hi[:]
        uj0[:] = uj[:] 
        t = t+dt
        if ( np.max(np.abs(Hi)) > 1000.0 or N < NSZ ):
            llwr=1
            break
 
       
 #   VOLT = np.sum(VolIn) / WaterVol[0]
 #   VOLF = WaterVol[i-1] / WaterVol[0]
 #   VOLP = (np.sum(VolIn) - (WaterVol[i-1] - WaterVol[0]) / WaterVol[0]
    # ----------------------------------------------------------------------------------------------------
 #   print( '%s %6.3f %s \n' % ('Total volume of water that has entered : Vin/V0 = ', 12.,'%' ) ) 
 #   print( '%s %6.3f %s \n' % ('Final volume of water in the domain    : Vf/V0  = ', 100*VOLF,'%' ) ) 
  #  print( '%s %6.3f %s \n' % ('Lost volume of water : (Vin-(Vf-V0))/V0 = ', 100*VOLP,'%' ) )
        
    set_interp = interp1d( Xj[0:Next+1], uj[0:Next+1], kind='cubic' )
    ui[0:Next] = set_interp(Xi[0:Next])    
       
    if full_output :
        for idx in range(len(ui)):
            fidUi.write( '%6.8f \t' % ui[idx] )
    
    ndt=len(Time)
    for elem in [H0, h0, T0, L0, dx, dt, Ntot, Tf, ndt, epsilon, kh0, slope ]:
        fidParam.write( '%6.8f \t' %   elem)
    fidParam.close()
    fidXj.close()
    fidFj.close()
    fidFxj.close()
    fidFxxj.close()
    fidHj.close()
    fidHi.close()
    fidUj.close()
    fidUi.close()
    fidUxj.close()
    fidUxxj.close()
    fidNUj.close()
    fidTime.close()
    fidNwet.close()
    fidXb.close()
    fidHout.close()



  #  fidParam=open('condlim.dat','w')
  #  for elem in [Time, a0,T0,L0,h0,Xlong,hLeft,uLeft,hRight,uRight,WaterVol,VolIn,dx,dt,g] :
  #      fidParam.write( '%6.8f \t' % elem)
  #  fidParam.close()
  #  np.save('debug',  hLeft)

   # llwr=false
    return  t #, llwr
  #  save(concat([carpetaN,'/condlim.mat']),'Time','a0','T0','L0','h0','Xlong','hLeft','uLeft','hRight','uRight','WaterVol','VolIn','dx','dt','g')
   # llwr=isempty(lastwarn)
#    return t,llwr
