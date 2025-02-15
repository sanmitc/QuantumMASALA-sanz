import numpy as np
import os
from qtm.config import qtmconfig
from qtm.constants import RYDBERG, ELECTRONVOLT
from qtm.containers.wavefun import get_WavefunG
from qtm.lattice import RealLattice
from qtm.crystal import BasisAtoms, Crystal
from qtm.pseudo import UPFv2Data
from qtm.kpts import gen_monkhorst_pack_grid, KList
from qtm.gspace import GSpace
from qtm.mpi import QTMComm
from qtm.dft import DFTCommMod, scf

from qtm.io_utils.dft_printers import print_scf_status

from qtm.logger import qtmlogger
from qtm.tddft_gamma.optical import dipole_response
# qtmconfig.fft_backend = 'mkl_fft'

DEBUGGING = True


if qtmconfig.gpu_enabled:
    qtmconfig.fft_backend = 'cupy'

from mpi4py.MPI import COMM_WORLD
comm_world = QTMComm(COMM_WORLD)
dftcomm = DFTCommMod(comm_world, comm_world.size, 1)

# Lattice
# print("WARNING!! : Please revert the alat back to 32 Bohr")
reallat = RealLattice.from_alat(alat=30.0, # Bohr
                                a1=[1., 0., 0.],
                                a2=[0., 1., 0.],
                                a3=[0., 0., 1.])


# Atom Basis
c_oncv = UPFv2Data.from_file('C_ONCV_PBE-1.2.upf')
h_oncv = UPFv2Data.from_file('H_ONCV_PBE-1.2.upf')

# C atom at the center of the cell
c_atoms = BasisAtoms.from_angstrom('C', c_oncv, 12.011, reallat,
                                  0.529177*np.array([15., 15., 15.]))
coords_ang = 0.642814093
h_atoms = coords_ang * np.array(
    [[ 1,  1,  1],
     [-1, -1,  1],
     [ 1, -1, -1],
     [-1,  1, -1]])
# Shift the H atoms to the center of the cell
h_atoms += 0.529177 * 15.0 * np.ones_like(h_atoms)
h_atoms = BasisAtoms.from_angstrom('H', h_oncv, 1.000, reallat,
                                  *h_atoms)



crystal = Crystal(reallat, [c_atoms, h_atoms])
kpts = KList.gamma(crystal.recilat)
print(kpts.numkpts)


# -----Setting up G-Space of calculation-----
ecut_wfn = 25 * RYDBERG
# NOTE: In future version, hard grid (charge/pot) and smooth-grid (wavefun)
# can be set independently
ecut_rho = 4 * ecut_wfn
gspc_rho = GSpace(crystal.recilat, ecut_rho)
gspc_wfn = gspc_rho

print("gspc_rho.reallat_dv", gspc_rho.reallat_dv)



is_spin, is_noncolin = False, False
numbnd = crystal.numel // 2
occ = 'fixed'
conv_thr = 1E-10 * RYDBERG
diago_thr_init = 1E-5 * RYDBERG

# occ = 'smear'
# smear_typ = 'gauss'
# e_temp = 1E-2 * RYDBERG


print('diago_thr_init :', diago_thr_init) #debug statement
# print('e_temp :', e_temp) #debug statement
print('conv_thr :', conv_thr) #debug statement
# print('smear_typ :', smear_typ) #debug statement
print('is_spin :', is_spin) #debug statement
print('is_noncolin :', is_noncolin) #debug statement
print('ecut_wfn :', ecut_wfn) #debug statement
print('ecut_rho :', ecut_rho) #debug statement


out = scf(dftcomm, crystal, kpts, gspc_rho, gspc_wfn,
        numbnd, is_spin, is_noncolin,
        occ_typ=occ,
        conv_thr=conv_thr, diago_thr_init=diago_thr_init,
        iter_printer=print_scf_status)

scf_converged, rho, l_wfn_kgrp, en = out

WavefunG = get_WavefunG(l_wfn_kgrp[0][0].gkspc, 1)


# if comm_world.rank == 0:

print("SCF Routine has exited")
print(qtmlogger)


# wfn_gamma = l_wfn_kgrp[0]

import numpy as np
from os import path, remove
for fname in ['rho.npy', 'wfn.npz']:
    if path.exists(fname) and path.isfile(fname):
        remove(fname)


from qtm.config import qtmconfig
# -----------------------
# BEGIN TDDFT CALCULATION
# -----------------------
gamma_efield_kick = 1e-4 # Electric field kick (in z-direction) in Hartree atomic units, 0.0018709241 Ry/e_Ry/Bohr = 0.01 Ha/e_Ha/Angstrom
time_step = 0.1    # Time in Hartree atomic units 1 Hartree a.u. = 2.4188843265864(26)×10−17 s. 
                    # Reference calculation (ce-tddft) had 2.4 attosecond time step.
numsteps = 10_002

qtmconfig.tddft_prop_method = 'etrs'
qtmconfig.tddft_exp_method = 'taylor'


# Pretty-print the input parameters for tddft
print("TDDFT Parameters:")
print("Electric field kick:", gamma_efield_kick)
print("Time step:", time_step)
print("Number of steps:", numsteps)
print("Propagation method:", qtmconfig.tddft_prop_method)
print("Exponential evaluation method:", qtmconfig.tddft_exp_method)
print(kpts.k_weights)

dip_z = dipole_response(comm_world, crystal, l_wfn_kgrp,
                        time_step, numsteps, gamma_efield_kick, 'z')


fname = 'dipz.npy'
if os.path.exists(fname) and os.path.isfile(fname):
    os.remove(fname)
np.save(fname, dip_z)

