
import numpy as np
import time
"""
This example file demonstrates the usage of G-space parallelization in QuantumMASALA.

The code performs a self-consistent field (SCF) calculation for a silicon supercell.

The main steps of the code are as follows:
1. Import necessary modules and libraries.
2. Set up the communication world for parallelization.
3. Define the lattice and atom basis for the crystal.
4. Generate the supercell based on the specified size.
5. Generate k-points using a Monkhorst Pack grid.
6. Set up the G-Space for the calculation.
7. Perform the SCF calculation using the specified parameters.
8. Print the SCF convergence status and results.

Example usage:
python si_scf_supercell.py <supercell_size>

Parameters:
- supercell_size: The size of the supercell in each dimension.

Output:
- SCF convergence status and results.

"""
from qtm.constants import RYDBERG
from qtm.lattice import RealLattice
from qtm.crystal import BasisAtoms, Crystal
from qtm.mpi.gspace import DistGSpace
from qtm.pseudo import UPFv2Data
from qtm.kpts import gen_monkhorst_pack_grid
from qtm.gspace import GSpace
from qtm.mpi import QTMComm
from qtm.dft import DFTCommMod, scf
from qtm.force import force, force_ewald, force_local, force_nonloc

from qtm.io_utils.dft_printers import print_scf_status

import argparse



from qtm import qtmconfig
from qtm.logger import qtmlogger

# qtmconfig.fft_backend = "pyfftw"
qtmconfig.set_gpu(False)

from qtm.config import MPI4PY_INSTALLED
if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

comm_world = QTMComm(COMM_WORLD)

# Only G-space parallelization
# K-point and/or band parallelization along with G-space parallelization is currently broken.
dftcomm = DFTCommMod(comm_world, 1, 1)

parser = argparse.ArgumentParser()
parser.add_argument("supercell_size", help="Side length of the supercell", type=int)

args = parser.parse_args()
supercell_size = args.supercell_size

alat=10.2*supercell_size
# Lattice
reallat = RealLattice.from_alat(
    alat, a1=[-0.5, 0.0, 0.5], a2=[0.0, 0.5, 0.5], a3=[-0.5, 0.5, 0.0]  # Bohr
)

def generate_coordinates(n):
    num_atoms = 2 * n**3
    min_distance = 1 / (4 * n)
    coordinates = []
    coordinates=np.zeros((num_atoms, 3))
    for inum in range(num_atoms):
        new_coord = np.random.uniform(-1, 1, 3)
        for iprev in range(inum):
            coord = coordinates[iprev]
            if np.linalg.norm(new_coord-coord) >= min_distance :
                coordinates[inum] = new_coord 
    return np.array(coordinates)

data= [
[0.00000000e+00,0.00000000e+00,0.00000000e+00],
[-1.26874010e-01,-4.79302744e-01,-6.10759530e-01],
[4.47320598e-01,3.56068727e-01,2.14939978e-01],
[7.35261283e-01,7.10631374e-01,-1.71979919e-01],
[5.95178903e-01,-2.99780556e-01,6.22027236e-01],
[3.21705506e-01,-6.33604345e-01,4.37190459e-01],
[-5.01662582e-01,-4.59022612e-01,-1.28027812e-01],
[4.61663485e-01,-5.61124891e-02,7.22458111e-04],
[-1.82235284e-01,1.24419831e-01,6.43527637e-01],
[6.31703350e-01,-6.59393935e-01,-7.15236381e-01],
[-6.01760002e-01,2.63670200e-01,-3.98291936e-01],
[-6.42023476e-01,-3.34217437e-01,-2.19091562e-01],
[6.75952042e-01,3.08814554e-01,-6.12721091e-01],
[-4.72891793e-02,1.89579113e-01,7.05939113e-02],
[4.91181274e-02,3.31468657e-01,1.22516957e-01],
[1.66439643e-01,-9.75885442e-02,-2.96180613e-01],
[-1.53434519e-01,6.47267116e-01,6.13048518e-01],
[7.82260133e-01,-2.58914723e-01,-6.91194930e-01],
[9.17270059e-02,3.85267675e-01,1.75586629e-01],
[5.16319462e-01,3.99666219e-01,-5.37132905e-01],
[1.23239508e-02,-1.63366082e-01,1.61514815e-01],
[3.55968704e-01,4.44123528e-01,6.33628585e-01],
[-6.38712643e-01,3.17482231e-02,5.05398707e-01],
[6.45656232e-01,7.57919233e-01,5.87418505e-01],
[-6.48687427e-01,-2.05626403e-01,2.85428698e-01],
[4.16671196e-02,4.12311225e-01,-5.23716825e-02],
[-3.59707128e-02,5.57446126e-03,-4.93162593e-01],
[-2.68437676e-01,3.61170634e-01,7.61734028e-01],
[-1.37912657e-01,2.03259373e-01,7.62316736e-01],
[2.95525483e-01,6.14261199e-01,5.29172992e-01],
[7.17401831e-01,5.62752441e-01,6.73835043e-01],
[-7.64983377e-01,-1.11698092e-01,-2.32283729e-01],
[-5.32217701e-01,-7.77385181e-01,-3.35421955e-01],
[5.09004378e-01,-2.90370382e-01,-3.24567542e-01],
[7.34674423e-01,2.04810486e-01,6.71424893e-01],
[4.45261665e-01,-6.07089439e-01,4.40525180e-01],
[7.06776717e-01,-6.81888855e-01,-9.76909291e-02],
[1.85421657e-01,-5.52565673e-01,4.78136379e-01],
[1.77263301e-01,6.26317314e-01,-2.23102596e-01],
[4.38386462e-01,-4.39912938e-01,-6.72237534e-01],
[4.19372251e-01,2.35118545e-01,-5.84648465e-03],
[3.70444245e-01,6.39836967e-01,4.14212345e-01],
[6.10804811e-01,-6.85500886e-01,6.69760035e-02],
[-1.06243850e-01,4.34830317e-01,-3.64171982e-01],
[1.19664058e-01,-2.56958474e-01,2.69747990e-01],
[-1.90562858e-01,1.14657245e-01,4.64925890e-01],
[4.97001563e-01,-2.97593832e-01,-3.77250708e-01],
[6.72481274e-01,2.16678025e-01,1.79163605e-01],
[5.37797379e-02,-2.94783523e-01,-7.03673869e-01],
[-4.20653006e-04,-6.08445985e-01,2.34014405e-01],
[3.93813294e-01,7.40272951e-01,-3.28358860e-01],
[-6.66284173e-02,1.95279253e-01,-2.85664231e-01],
[5.01856874e-01,-1.33779730e-01,-4.53433005e-01],
[4.16701642e-01,6.61833568e-01,-4.55031196e-01]]

N=np.array(data)
# Atom Basis
si_oncv = UPFv2Data.from_file("Si_ONCV_PBE-1.2.upf")
si_atoms = BasisAtoms.from_alat(
    "si",
    si_oncv,
    28.086,
    reallat,
    N,
)

crystal = Crystal(reallat, [si_atoms])  # Represents the crystal

#crystal = crystal.gen_supercell([supercell_size] * 3)
si_basis=crystal.l_atoms[0]
##We want to print the coordinates of the Si atms
#print("Si basis", si_basis.r_alat)

#coordinates=generate_coordinates(supercell_size).T
## Set this as the new coordinates of the basis
#si_basis.r_cart=coordinates


#print("new coordinates", si_basis.r_cart)



# Generating k-points from a Monkhorst Pack grid (reduced to the crystal's IBZ)
mpgrid_shape = (1, 1, 1)
mpgrid_shift = (False, False, False)
kpts = gen_monkhorst_pack_grid(crystal, mpgrid_shape, mpgrid_shift)

# -----Setting up G-Space of calculation-----
ecut_wfn = 15 * RYDBERG
ecut_rho = 4 * ecut_wfn
grho_serial = GSpace(crystal.recilat, ecut_rho)

# If G-space parallelization is not required, use the serial G-space object
#print("N_pwgrp", dftcomm.n_pwgrp)
#print("Image_comm_size", dftcomm.image_comm.size)
if dftcomm.n_pwgrp == dftcomm.image_comm.size:  
    grho = grho_serial
else:
    grho = DistGSpace(comm_world, grho_serial)
gwfn = grho

print("the type of grho is", type(grho))    

numbnd = crystal.numel // 2  # Ensure adequate # of bands if system is not an insulator
conv_thr = 1e-8 * RYDBERG
diago_thr_init = 1e-2 * RYDBERG

out = scf(
    dftcomm,
    crystal,
    kpts,
    grho,
    gwfn,
    numbnd,
    is_spin=False,
    is_noncolin=False,
    symm_rho=True,
    rho_start=None,
    occ_typ="smear",
    smear_typ="gauss",
    e_temp=0.01 * RYDBERG,
    conv_thr=conv_thr,
    diago_thr_init=diago_thr_init,
    iter_printer=print_scf_status,
    force_stress=True
)

scf_converged, rho, l_wfn_kgrp, en, v_loc, nloc, xc_compute= out



start_time = time.time()
force_ewa=force_ewald(dftcomm=dftcomm,
                      crystal=crystal,
                      gspc=gwfn, 
                      gamma_only=False)

if dftcomm.image_comm.rank==0:
    print("force ewald", force_ewa)
    print("Time taken for ewald force: ", time.time() - start_time)
print(flush=True)

##Calculation time of Local Forces
start_time = time.time()
force_loc=force_local(dftcomm=dftcomm,
                      cryst=crystal, 
                      gspc=gwfn, rho=rho, 
                      vloc=v_loc,
                      gamma_only=False)

if dftcomm.image_comm.rank==0:
    print("force local", force_loc)
    print("Time taken for local force: ", time.time() - start_time)
print(flush=True)

##Calculation time of Non Local Forces
start_time = time.time()
force_nloc=force_nonloc(dftcomm=dftcomm,
                          numbnd=numbnd,
                          wavefun=l_wfn_kgrp, 
                          crystal=crystal,
                          nloc_dij_vkb=nloc)

if dftcomm.image_comm.rank==0:
    print("force non local", force_nloc)
    print("Time taken for non local force: ", time.time() - start_time)
print(flush=True)

#force_time=time.time()
force_total, force_norm=force(dftcomm=dftcomm,
                            numbnd=numbnd,
                            wavefun=l_wfn_kgrp,
                            crystal=crystal,
                            gspc=gwfn, 
                            rho=rho,
                            vloc=v_loc,
                            nloc_dij_vkb=nloc,
                            gamma_only=False,
                            verbosity=True)



if comm_world.rank == 0:
    print("SCF Routine has exited")
    print(qtmlogger)
