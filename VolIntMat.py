import numpy as np 
from scipy import sparse as sps
#from numba import jit

    
#@jit
def VolIntMat(Ntot=None):

    # Assembling Volume Integration Matrix for j cells using 4th order approximation
# Simpson Rule and cubic interpolation for cell face values
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Arguments :
#  Ntot : Total number of cell faces (j nodes)
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Internal nodes

    row = np.arange(2,Ntot-2)
    col = np.arange(0,Ntot-4)
    data = np.zeros(Ntot-4)
    data[:]=-1./96.
    A1=sps.csr_matrix((data, (row, col)), shape=(Ntot, Ntot))

    row = np.arange(2,Ntot-2)
    col = np.arange(1,Ntot-3)
    data = np.zeros(Ntot-4)
    data[:]=8./96.
    A2=sps.csr_matrix((data, (row, col)), shape=(Ntot, Ntot))

    row = np.arange(2,Ntot-2)
    col = np.arange(2,Ntot-2)
    data = np.zeros(Ntot-4)
    data[:]=82./96.
    Ad=sps.csr_matrix((data, (row, col)), shape=(Ntot, Ntot))

    row = np.arange(2,Ntot-2)
    col = np.arange(3,Ntot-1)
    data = np.zeros(Ntot-4)
    data[:]=8./96.
    A3=sps.csr_matrix((data, (row, col)), shape=(Ntot, Ntot))

    row = np.arange(2,Ntot-2)
    col = np.arange(4,Ntot)
    data = np.zeros(Ntot-4)
    data[:]=-1./96.
    A4=sps.csr_matrix((data, (row, col)), shape=(Ntot, Ntot))


    Aj=A1 + A2 + Ad + A3 + A4
    
    # Boundary nodes
    Aj[0,0:4]=1./ 96*np.array([104,- 20,16,- 4])
    Aj[1,0:3]=1./96.*np.array([4,88,4])
    Aj[Ntot-2,Ntot-3:Ntot]=1./96.*np.array([4,88,4])
    Aj[Ntot-1,Ntot-4:Ntot]=1./96.*np.array([- 4,16,- 20,104])
    # ----------------------------------------------------------------------------------------------------
    return Aj
