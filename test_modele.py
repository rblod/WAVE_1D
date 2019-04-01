from modele import modele
from os import getcwd
import matplotlib.pyplot as plt
import numpy as np
##
do_plot=False

niter = 5
RootFolder = getcwd()
tfin  = 360


# Bathy and wave
rd_bathy    = True    # random bathy
rd_wave     = True    # random waves with constant energy 
rd_wave_tot = True   # random energy (in conjonction with rd_wave=True)

# Time stack generation : 
nstart = 60             # time cropped (s)
newdx  = 2.            # spatial resolution (m)
nfps   = 2.            # temporal resolution(Hz) 
withnoise = True
noisemin = 0.25
noisemax = 0.75

       
for i in range(niter):

 	TimeStack, Zprof, nn = modele( RootFolder, tfin    , i          , 
                                   rd_bathy  , rd_wave , rd_wave_tot, 
                                   nstart    , nfps    , newdx , 
                                   withnoise , noisemin, noisemax  )

 	if do_plot:
		fig, (ax1,ax2,ax3) = plt.subplots(3,1)
		vmin = np.min(np.array([TimeStack,TimeStack*nn]))
		vmax = np.max(np.array([TimeStack,TimeStack*nn]))

		c = ax1.pcolormesh(TimeStack, cmap='RdBu')
		ax1.set_title('TimeStack')
		c.set_clim(vmin,vmax)
		fig.colorbar(c, ax=ax1)
		c = ax2.pcolormesh(TimeStack*nn, cmap='RdBu')
		ax2.set_title('Noisy TimeStack')
		c.set_clim(vmin,vmax)
		fig.colorbar(c, ax=ax2)
		c = ax3.pcolormesh(nn, cmap='RdBu')
		ax3.set_title('Nois')
		fig.colorbar(c, ax=ax3)
		c.set_clim(0.,1.)
		plt.tight_layout()
		plt.savefig(str(i)+'_stack.png')
		plt.close()

#	plt.show()
		fig, (ax2,ax3,ax4) = plt.subplots(3,1)
		ax2.plot(Zprof)
		ax2.grid(True)
		ax2.set_title('Bathy')
		myt=int(TimeStack.shape[0])
		ax3.plot(TimeStack[int(myt/2.),:])
		ax3.plot(TimeStack[myt-1,:])
		ax3.grid(True)
		ax3.set_title('TimeStack')
		ax4.plot(TimeStack[:,0])
		ax4.grid(True)
		ax4.set_title('Wave Forcing')
		plt.tight_layout()
		plt.savefig(str(i)+'_wave.png')
		plt.close()
	#plt.savefig(str(niter)+'_zprof.png')


