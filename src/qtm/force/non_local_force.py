import numpy as np
from qtm.crystal import Crystal
from qtm.pseudo.nloc import NonlocGenerator
from qtm.constants import RYDBERG_HART
from qtm.config import NDArray
from qtm.dft import DFTCommMod
from qtm.mpi import scatter_slice

from time import perf_counter

def force_nonloc(dftcomm:DFTCommMod,
                 numbnd: int,
                 wavefun: tuple,
                 crystal: Crystal,
                 nloc_dij_vkb:list
                 ) -> NDArray:
    """Calculate the nonlocal force on the atoma from the upf potential"""

    ##Starting of the parallelization over bands
    assert isinstance(numbnd, int)

    with dftcomm.kgrp_intra as comm:
        band_slice=scatter_slice(numbnd, comm.size, comm.rank)

    ##Getting the characteristics of the crystal
    #start_time=perf_counter()
    l_atoms = crystal.l_atoms
    num_typ=len(l_atoms)
    tot_atom = np.sum([sp.numatoms for sp in l_atoms])
    #labels=np.repeat(np.arange(len(l_atoms)), [sp.numatoms for sp in l_atoms])
    atom_label= np.concatenate([np.arange(sp.numatoms) for sp in l_atoms])

    ##Initializing the force array 
    force_nl=np.zeros((tot_atom, 3))
    #if dftcomm.image_comm.rank==0: print("initializing the force array", perf_counter()-start_time)

    k_counter=0
    for wfn in wavefun:
        for k_wfn in wfn:
            #k_loop_start=perf_counter()
            ## Getting the evc and gk-space characteristics form the wavefunction
            #start_time=perf_counter()
            evc=k_wfn.evc_gk[band_slice]
            gkspace=k_wfn.gkspc
            k_weight=k_wfn.k_weight
            gkcart=gkspace.gk_cart.T
            #k_cryst=k_wfn.k_cryst
            #k_tpiba=k_wfn.gkspc.recilat.cryst2tpiba(k_cryst)
            dij_vkb=nloc_dij_vkb[k_counter]

            ##Getting the occupation number
            occ_num=k_wfn.occ[band_slice]
            #if dftcomm.image_comm.rank==0: print(f"In {k_counter}th k loop, initialization time", perf_counter()-start_time)
            atom_counter=0
            ## Getting the non-local beta projectors and dij matrices from the wavefun
            ##It is being calculated in the root and then broadcasted over to other processors in same k group according to band parallelization
            for ityp in range(num_typ):
                sp=l_atoms[ityp]
                #start_time=perf_counter()
                #k_nonloc= NonlocGenerator(sp=sp,
                #                            gwfn=gkspace.gwfn)
                #vkb_full, dij, vkb_diag = k_nonloc.gen_vkb_dij(k_wfn.gkspc)
                vkb_full, dij = dij_vkb[ityp]
                #if dftcomm.image_comm.rank==0: print(f"In {k_counter}th k loop, nonlocal generator time", perf_counter()-start_time)

                for inum in range(sp.numatoms):
                    #inum_loop_start=perf_counter()
                    atom_label_sp=atom_label[inum]

                    ##Getting the non-local force
                    #start_time=perf_counter()
                    row_vkb=int(vkb_full.data.shape[0]/sp.numatoms)
                    vkb=vkb_full.data[atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb]
                    dij_sp=dij[atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb, atom_label_sp*row_vkb:(atom_label_sp+1)*row_vkb]
                    #if dftcomm.image_comm.rank==0: print(f"In {k_counter}th k loop, getting the vkb and dij time", perf_counter()-start_time)

                    ##Constructing the G\beta\psi
                    #start_time=perf_counter()
                    gkcart_struc=gkcart.reshape(-1,1,3)
                    evc_sp=evc.data.T            
                    Kc=gkcart_struc*evc_sp[:,:,None]
                    GbetaPsi=np.einsum("ij, jkl->ikl", vkb, np.conj(Kc))
                    #if dftcomm.image_comm.rank==0: print(f"In {k_counter}th k loop, constructing GbetaPsi time", perf_counter()-start_time)

                    ##Constructing the \beta\psi
                    #start_time=perf_counter()
                    betaPsi=np.conj(vkb)@(evc_sp*occ_num.reshape(1,-1))
                    betaPsi=dij_sp@betaPsi
                    #if dftcomm.image_comm.rank==0: print(f"In {k_counter}th k loop, constructing betaPsi time", perf_counter()-start_time)

                    ##Multiplying Together
                    #start_time=perf_counter()

                    V_NL_Psi=np.sum(GbetaPsi*betaPsi.reshape(*betaPsi.shape, 1), axis=(0,1))
                    trace=-2*np.imag(V_NL_Psi)
                    #if dftcomm.image_comm.rank==0: print(f"In {k_counter}th k loop, multiplying together time", perf_counter()-start_time)

                    ##Multiply by Weight
                    #start_time=perf_counter()
                    trace = trace * k_weight
                    force_nl[atom_counter]+=trace 

                    atom_counter+=1
            #if dftcomm.image_comm.rank==0: print(f"In {k_counter}th k loop, multiplying by weight time", perf_counter()-start_time)
            #if dftcomm.image_comm.rank==0: print("time for 1 k loop", perf_counter()-k_loop_start)
        k_counter+=1
    force_nl/=RYDBERG_HART
    with dftcomm.image_comm as comm:
        comm.Allreduce(comm.IN_PLACE, force_nl)
        #if comm.rank==0: if dftcomm.image_comm.rank==0: print("nonlocal force before symmetrization", force_nl)  
        #if comm.rank==0: if dftcomm.image_comm.rank==0: print( "nonlocal force after symmetrization", force_nl)
    #force_nl=crystal.symm.symmetrize_vec(force_nl)
    return force_nl