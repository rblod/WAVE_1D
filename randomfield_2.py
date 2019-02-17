import numpy as np
from random import random as rand
from math import pi, cos, sin, sqrt


    #Noise
# for i=1:4
# ran = randomfield_2([1000 2048],[1 1],i*[100 10],80);
# eval(['noise.n',num2str(i),'=ran']);
# figure(74);subplot(2,2,i);imagesc(ran);
# end
# noise.info='randomfield_2([1000 2048],[1 1],i*[100 10],80) with i=1:4 (19 Jan 2017)'
# load('Depth_Inversion_noise','noise')
    
    #Rampe
# v=linspace(0,1,size(ran,2));
# v2=repmat(v,size(S,1));
    
    # Changer le niveau de bruit
# level=[0.4 0.5 0.6 1];
# for i=1:4
# for h=1:length(level)
    
    # eval(['nn=((noise.n',num2str(i),')./4+1)./2;']);
#     ii=find(nn>=level(h));nn(ii)=0;
    
    # ratio=round(100.*(length(ii)./length(nn(:))))
# #   figure(87);subplot(1,2,1);imagesc(S(:,:).*nn(:,1:size(S,2)))
# #   subplot(1,2,2);imagesc(nn)
# #Rampe
# # v=linspace(0,1,size(ran,2));
# # v2=repmat(v,size(S,1));
    
    # eval(['Noise.n',num2str(i),'.ratio',num2str(ratio),'=nn']);
# end
# end
    
    # save('Depth_Inversion_Noise','Noise')
    
    # Noise.info=['randomfield_2([1000 2048],[1 1],i*[100 10],80) with i=1:4 (19 Jan 2017), level of noise level=[0.4 0.5 0.6 1];nn=((noise.n4)./4+1)./2;  ii=find(nn>=level(h));nn(ii)=0;ratio=round(100.*(length(ii)./length(nn(:)))) '];
# save('Depth_Inversion_Noise','Noise')
    
    
def randomfield_2(nx0=None,dx0=None,lt0=None,ang0=None):

    # This function has no output args, see OutputFcn.
# hObject    handle to figure
# eventdata  reserved - to be defined in a future version of MATLAB
# handles    structure with handles and user data (see GUIDATA)
# varargin   command line arguments to randomfield (see VARARGIN)
  
 #   global nx,dx,lx,ang,dimen,Ctype,sigma2,X,Y,Z,RYY,Rotmat,i_real,mu
    nx = np.array([nx0[0],nx0[1]])
    dx =np.array([dx0[0],dx0[1]])
    lx = np.array([lt0[0],lt0[1]])
    ang = np.array([ang0,ang0])
    Ctype=1
    dimen=2
    sigma2=1
    mu=0

    #   Copyright 2017 Rafael Almar (IRD, France)- rafael.almar@ird.fr
# Choose default command line output for randomfield
# handles.output = hObject;
# 
# # Update handles structure
# guidata(hObject, handles);
    
    # This sets up the initial plot
    RYY = calc_covar(nx,dx,ang,lx,sigma2)
 
    ang2=ang / 180.*pi
        
    X,Y  = np.meshgrid(np.arange( -nx[1]/2.*dx[1], (nx[1]-1)/2.*dx[1], dx[1] ),
                       np.arange( -nx[0]/2.*dx[0], (nx[0]-1)/2.*dx[0], dx[0] ) )
    X2 =  cos(ang2[0])*X + sin(ang2[0])*Y
    Y2 = -sin(ang2[0])*X + cos(ang2[0])*Y
    H = np.sqrt( (X2 / lx[0]) ** 2 + (Y2 / lx[1]) ** 2)
#    global nx,dx,lx,ang,dimen,Ctype,sigma2,X,Y,Z,RYY,Rotmat,i_real,ran,mu
    ntot = np.prod(nx[0:dimen])
    n_real=1
    # ============== BEGIN POWER-SPECTRUM BLOCK =======================================
# Fourier Transform (Origin Shifted to Node (1,1))
# Yields Power Spectrum of the field

    SYY = np.fft.fftn(np.fft.fftshift(RYY)) / ntot

    # Remove Imaginary Artifacts
    SYY = abs(SYY)
    SYY[0,0]=0
    # ============== END POWER-SPECTRUM BLOCK =========================================
    
    # ============== BEGIN FIELD GENERATION BLOCK =====================================
# Generate the field
 #   i_real = i_real + 1
    # nxhelp is nx with the first two entries switched
    nxhelp = nx[0:dimen]
    if (dimen > 1):
        nxhelp[0:2] = np.array( [ nx[0],nx[1] ] )
    else:
        nxhelp = np.array( [ nx[0],1 ])
    
    # Generate a field of random real numbers,
# transform the field into the spectral domain,
# and evaluate the corresponding phase-spectrum.
# This random phase-spectrum and the given power spectrum
# define the Fourier transform of the random autocorrelated field.
    ran = np.sqrt(SYY)*np.exp( complex(0,1)*np.angle( np.fft.fftn( np.random.rand(nxhelp[0],nxhelp[1]) ) ) )
    # Backtransformation into the physical coordinates
    ran = np.real(np.fft.ifftn(ran*ntot)) + mu
    return ran
    
if __name__ == '__main__':
    pass
    
    # --- Calculates the covariance function.
    
def calc_covar(nx,dx,ang,lx,sigma2):

 #   global nx,dx,lx,ang,dimen,Ctype,sigma2,X,Y,Z,Rotmat
    ntot = np.prod(nx)
    ang2=ang/180.*pi
    
    X,Y  = np.meshgrid(np.arange( -nx[1]/2.*dx[1], (nx[1]-1)/2.*dx[1], dx[1] ),
                       np.arange( -nx[0]/2.*dx[0], (nx[0]-1)/2.*dx[0], dx[0] ) )
   
    X2 =  cos(ang2[0])*X + sin(ang2[0])*Y
    Y2 = -sin(ang2[0])*X + cos(ang2[0])*Y
    H  =  np.sqrt( (X2/lx[0]) ** 2 + (Y2 / lx[1] ) ** 2)
    RYY = np.exp(- H ** 2)*sigma2
    
    return RYY
    
if __name__ == '__main__':
    pass
    