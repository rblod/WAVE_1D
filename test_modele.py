from modele import modele
import os
import matplotlib.pyplot as plt
import numpy as np
##

niter = 20


RootFolder = os.getcwd()
tfin  = 720


# Bathy and wave
rd_bathy    = True    # random bathy
rd_wave     = True    # random waves with constant energy 
rd_wave_tot = False   # random energy (in conjonction with rd_wave=True)

# Time stack generation : 
nstart = 60            # time cropped (s)
newdx  = 2.            # spatial resolution (m)
nfps   = 2.            # temporal resolution(Hz) 
withnoise = False
noisemin = 0.25
noisemax = 0.75

       
for i in range(niter):

 	TimeStack, Zprof, nn = modele( RootFolder, tfin , i, 
                               rd_bathy  , rd_wave, rd_wave_tot, 
                               nstart    , nfps   , newdx , 
                               withnoise , noisemax  )

	fig, (ax1,ax2) = plt.subplots(1,2)
	c = ax1.pcolor(TimeStack, cmap='RdBu')
	ax1.set_title('TimeStack')
	fig.colorbar(c, ax=ax1)
	c = ax2.pcolor(TimeStack*nn, cmap='RdBu')
	ax2.set_title('Noisy TimeStack')
	fig.colorbar(c, ax=ax2)
#	plt.show()
	plt.savefig(str(niter)+'_foo.png')
	plt.plot(Zprof)
	plt.savefig(str(niter)+'_zprof.png')


