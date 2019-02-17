import os
##from numba import jit

##@jit
def opfiles(carpeta=None,RootFolder=None,*args,**kwargs):


    # ----------------------------------------------------------------------------------------------------
    fidParam=open(carpeta+'/param.dat','w')
    
    fidXj=open(carpeta+'/Xj.dat','w')
    
    fidXi=open(carpeta+'/Xi.dat','w')
    
    fidFj=open(carpeta+'/Fj.dat','w')
    
    fidFxj=open(carpeta+'/Fxj.dat','w')
    
    fidFxxj=open(carpeta+'/Fxxj.dat','w')
    
    fidHj=open(carpeta+'/Hj.dat','w')
    
    fidHi=open(carpeta+'/Hi.dat','w')
    
    fidUj=open(carpeta+'/Uj.dat','w')
    
    fidUi=open(carpeta+'/Ui.dat','w')
    
    fidUxj=open(carpeta+'/Uxj.dat','w')
    
    fidUxxj=open(carpeta+'/Uxxj.dat','w')
    
    fidNUj=open(carpeta+'/NUj.dat','w')
    
    fidTime=open(carpeta+'/time.dat','w')
    
    fidNwet=open(carpeta+'/Nwet.dat','w')
    
    fidXb=open(carpeta+'/Xb.dat','w')
    
    fidHout=open(carpeta+'/Hout.dat','w')
    
    # ----------------------------------------------------------------------------------------------------
    
    return fidParam,fidXj,fidXi,fidFj,fidFxj,fidFxxj,fidHj,fidHi,fidUj,fidUi,fidUxj,fidUxxj,fidNUj,fidTime,fidNwet,fidXb,fidHout
