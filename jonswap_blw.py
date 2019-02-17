import numpy as np
from numpy import dot,sqrt,multiply,cos,sin,tanh,cosh,sinh
from numpy.random import random_sample
from math import pi
#from numba import njit

    
#@jit
def jonswap_blw(duree=None,freq=None,fp=None,Hmo=None,nfre=None,gama=None,hh=None,rd_wave=None):

#    Tp1=10
#    a1=0.5
#    dt=0.01
#    duree=1000
#    freq=int(1. / dt)
#    fp=1. / Tp1
#    Hmo=a1
#    nfre=100
#    gama=3.
#    hh=3.


    ## [ff,SSm,tt,eta,etac,etaLB]=jonswap_blw(1200,50,.5,.16,100,3.3,.553);
## freq pic / Hmo = 1.416 * Hrms / nb de bandes en freq
## cf Goda (1985) p.26
    
    #   Copyright 2017 Rafael Almar (IRD, France)- rafael.almar@ird.fr
# compensation bound wave
    
    # INPUT:
# duree = duree du signal que l'on souhaite genere
# freq = frequence d'echantillonage
# fp = frequence pic du spectre de jonswap
# Hm0 = hauteur significative
# nfre = nombre de composantes
# gama = facteur de pic
# hh= profondeur d'eau au batteur
    
    #OUTPUT:
# ff = vecteur des frequences de chaque composante
# SSm = la DSP
# tt = vecteur temps de duree "duree" de resolution "freq"
# eta = deplacement partie jonswap de la surface libre
# etac = deplacement de la surface libre total (htes freq + bound wav)
# etaLB= bound wave seule
    
    # gama=3.3; ## freq standard (1<gama<7 - essayer aussi gama=200)
    siga=0.07
    sigb=0.09
    gg=9.8
    #---------------------------------------------------------------------
# decoupage bande de frequence
    fmin = 0.5*fp
    fmax = 5*fp
    df=(fmax - fmin) / nfre
    #ff1=[fmin:df:fp];
    ff1=np.arange(fp,fmin, -df)

    if rd_wave:
        ff1=np.sort(ff1) + dot((random_sample(len(ff1)) - 0.5),df)
    else:    
        ff1=np.sort(ff1) + dot((np.arange(1,len(ff1)+1,1) - 0.5),df)
    ff2=np.arange(fp + df,fmax,df)
    if rd_wave:
        ff2=ff2 + dot((random_sample(len(ff2)) - 0.5),df)
    else:    
        ff2=ff2 + dot((np.arange(1,len(ff2)+1,1) - 0.5),df)
    ff=np.concatenate([ff1,ff2])
#    print len(ff)
    
    #-----------------------------------------------------------------------
# calcul du spectre
    Gf1=gama ** np.exp(- (ff1 / fp - 1) ** 2.0 / 2 / siga ** 2)
    Gf2=gama ** np.exp(- (ff2 / fp - 1) ** 2.0 / 2 / sigb ** 2)
    Gf=np.concatenate([Gf1,Gf2])
    af=dot(dot(dot(0.0624,Hmo),Hmo),fp ** 4) / (0.23 + dot(0.0336,gama) - 0.185 / (1.9 + gama))
    SSm=multiply(dot(Gf,af) / ff ** 5,np.exp((-5./4.*(ff/fp)**-4)))
    Hm0_m_cons=dot(4,sqrt(dot(np.sum(SSm),df)))
    
    # figure,plot(ff,SS,ff,SSm,'--')
#-------------------------------------------------------------------------
# determination de chaq composantes
    AA=sqrt(2*SSm*df)
    
 #RB   phi=dot(dot(2,pi),random_sample(nfre))
    phi=2*pi*np.arange(1,nfre+1,1)
    
    tt=np.arange(0,duree+1./freq,1. / freq)
    eta=np.zeros(len(tt))
    #-------------------------------------------------------------------------
# recomposition de la surface libre au premier ordre
    kk=np.zeros(nfre)
#    print AA.shape,  nfre
    #    print tt.shape
#    print ff.shape
#    print phi.shape
#    return

    for ii in range(0,nfre):
        etf=dot(AA[ii],cos(multiply(dot(dot(2,pi),tt),ff[ii]) + phi[ii]))
        eta=eta + etf
        kk[ii]=dot(dot(2,pi),ff[ii]) / sqrt(dot(gg,hh))
        for it in range(0,30):
            kk[ii]=(dot(dot(2,pi),ff[ii])) ** 2 / (dot(gg,tanh(dot(kk[ii],hh))))
    
    #-------------------------------------------------------------------------
# Determination de la bound wave
    Dm=dot(ff,sqrt(hh / gg))
    aan=multiply(AA,cos(phi))
    bbn=multiply(- AA,sin(phi))
    
    #le nlim permet d'eliminer les paires de composantes qui donnent des hautes freq >fmin
    nlim=int(round(nfre / 9.))
    
    etaLB=0
    for im in range(0,nfre-1):
        som=0
        #    for in=im+1:nfre
        for in_ in range(im+1,nlim):
            dkmn=kk[in_] - kk[im]
            dff=ff[in_] - ff[im]
            G1=dot(dot(dot(dot(dot(dot(4,pi ** 2),Dm[im]),Dm[in_]),dkmn),hh),cosh(dot(dkmn,hh))) / (cosh(dot((kk[im] + kk[in_]),hh)) - cosh(dot(dkmn,hh)))
            G2=(dot(dot(dot(dot(dot(dkmn,hh),(Dm[in_] - Dm[im])),hh),(dot(kk[in_],Dm[im]) + dot(kk[im],Dm[in_]))),1./tanh(dot(dkmn,hh)))) / 2 / Dm[in_] / Dm[im]
            G3=dot(dot(dot(dot(2,pi ** 2),(Dm[in_] - Dm[im]) ** 2),dkmn),hh)
            Gmn=(G1 + G2 - G3) / hh / (dot(dot(dot(4,pi ** 2),(Dm[in_] - Dm[im]) ** 2),1./tanh(dot(dkmn,hh))) - dot(dkmn,hh))
            t1mn=dot(aan[in_],aan[im]) + dot(bbn[in_],bbn[im])
            t2mn=dot(aan[im],bbn[in_]) - dot(bbn[im],aan[in_])
            etamn2=dot(Gmn,(dot(t1mn,cos(dot(dot(dot(2,pi),dff),tt))) + dot(t2mn,sin(dot(dot(dot(2,pi),dff),tt)))))
            som=som + etamn2
        etaLB=etaLB + som
        
    etac=eta + etaLB
    return ff,SSm,tt,eta,etac,etaLB
