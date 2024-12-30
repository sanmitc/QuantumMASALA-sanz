import numpy as np


from qtm.crystal import Crystal
from qtm.gspace import GSpace
from qtm.containers.field import FieldGType
from qtm.pseudo.loc import loc_generate_pot_rhocore
from qtm.constants import RYDBERG,PI
from qtm.dft import DFTCommMod

from time import perf_counter

##RYDBERG=1/2
###### The energy scale is in RYDbergs all the potential are half than Quantum Espresso.
def force_local(dftcomm: DFTCommMod,
                cryst: Crystal, 
                gspc: GSpace, 
                rho:FieldGType,
                vloc:list,
                gamma_only:bool=False):

    #Setting up characteristics of the crystal
    #start_time=perf_counter()
    l_atoms=cryst.l_atoms
    numatoms=[sp.numatoms for sp in l_atoms]
    tot_num = np.sum(numatoms)
    num_typ=len(l_atoms)
    labels=np.repeat([np.arange(num_typ)], [sp.numatoms for sp in l_atoms])
    coords_cart_all = np.concatenate([sp.r_cart for sp in l_atoms], axis=1)
    rho=rho._data[0, rho.gspc.idxsort]/np.prod(rho.gspc.grid_shape)

    #setting up G space characteristics
    idxsort = gspc.idxsort
    numg = gspc.size_g
    cart_g = gspc.g_cart[:,idxsort]
    if cart_g.ndim==3:
        cart_g=cart_g.reshape(cart_g.shape[0],cart_g.shape[-1])
    print("coords_cart_all_shape", coords_cart_all.shape)
    print("cart_g_shape", cart_g.shape)
    gtau = coords_cart_all.T @ cart_g
    omega=gspc.reallat_cellvol
    #if dftcomm.image_comm.rank==0: print('Time to setup the crystal and g-space', perf_counter()-start_time)

    #start_time=perf_counter()
    ###Constructing the local pseudopotential
    v_loc=np.zeros((tot_num, numg))
    for isp in range(num_typ):
        v_loc[labels==isp]=vloc[isp].data
    v_loc=v_loc[:,idxsort]
 
    #if dftcomm.image_comm.rank==0: print('Time to construct the local pseudopotential', perf_counter()-start_time)

    #start_time=perf_counter()
    fact=2 if gamma_only else 1
    force_local=np.zeros((tot_num, 3))
    vrho=np.multiply(v_loc,(np.imag(np.exp(1j*gtau)*rho)/RYDBERG))
    #if dftcomm.image_comm.rank==0: print("time taken to calculate the vrho", perf_counter()-start_time)
    #start_time=perf_counter()
    force_local=vrho@cart_g.T*omega*fact
    print("force_local", force_local)   
    #if dftcomm.image_comm.rank==0: print("time taken to calculate the force through matrix multiplications", perf_counter()-start_time)
    #start_time=perf_counter()
    force_local=cryst.symm.symmetrize_vec(force_local[0] if cart_g.ndim==3 else force_local)
    #if dftcomm.image_comm.rank==0: print("time taken to symmetrize the force", perf_counter()-start_time)
    return force_local