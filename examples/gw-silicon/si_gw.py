
# Tutorial: G0W0 Approximation 
# 
# In this example, we present an example calculation of quasiparticle energies using QuatumMASALA's `gw` module.


# Imports
import numpy as np
import sys

sys.path.append(".")

dirname = "./"


# DFT Calculation
# We will start with a DFT calculation to get the energy eigenfunctions and eigenvalues.


import numpy as np

from qtm.constants import RYDBERG, ELECTRONVOLT
from qtm.kpts import KList
from qtm.lattice import RealLattice
from qtm.crystal import BasisAtoms, Crystal
from qtm.pseudo import UPFv2Data
from qtm.gspace import GSpace
from qtm.mpi import QTMComm
from qtm.dft import DFTCommMod, scf

from qtm.io_utils.dft_printers import print_scf_status

from qtm import qtmconfig
from qtm.logger import qtmlogger
qtmconfig.set_gpu(False)
# qtmconfig.fft_backend = 'mkl_fft'

from qtm.config import MPI4PY_INSTALLED
if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None
comm_world = QTMComm(COMM_WORLD)
dftcomm = DFTCommMod(comm_world, 1, comm_world.size)    
# FIXME: kpts.KList and klist.KList are not fully compatible.
#        Therefore, as a temporary fix, we are running the dft calculations serially.

# Lattice
reallat = RealLattice.from_alat(alat=10.2,  # Bohr
                                a1=[-0.5,  0. ,  0.5],
                                a2=[ 0. ,  0.5,  0.5],
                                a3=[-0.5,  0.5,  0. ])

# Atom Basis
si_oncv = UPFv2Data.from_file('Si_ONCV_PBE-1.2.upf')
si_atoms = BasisAtoms.from_alat('Si', si_oncv, 28.086, reallat,
                               np.array([[0.875, 0.875, 0.875], [0.125, 0.125, 0.125]]))

crystal = Crystal(reallat, [si_atoms, ])  # Represents the crystal


# Generating k-points from a Monkhorst Pack grid (reduced to the crystal's IBZ)
mpgrid_shape = (4, 4, 4)
x = np.linspace(0,1,mpgrid_shape[0], endpoint=False)
y = np.linspace(0,1,mpgrid_shape[1], endpoint=False)
z = np.linspace(0,1,mpgrid_shape[2], endpoint=False)
xx,yy,zz = np.meshgrid(x,y,z, indexing="ij")
kcryst = np.vstack([xx.flatten(),yy.flatten(),zz.flatten()])
kpts = KList(recilat=crystal.recilat, k_coords=kcryst, k_weights=np.ones(kcryst.shape[1])/kcryst.shape[1])


# -----Setting up G-Space of calculation-----
ecut_wfn = 25 * RYDBERG
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal.recilat, ecut_rho)
gwfn = grho

# -----Spin-polarized (collinear) calculation-----
is_spin, is_noncolin = False, False
mag_start = None#[0.0]
numbnd_occ = 4
numbnd_nscf = 30

occ = 'fixed'

conv_thr = 1E-8 * RYDBERG
diago_thr_init = 1E-2 * RYDBERG


# ### DFT: SCF calculation for occupied bands


from qtm.kpts import KList


kpts = KList(recilat=crystal.recilat, k_coords=kpts.k_cryst, k_weights=np.ones(kpts.k_cryst.shape[1])/kpts.k_cryst.shape[1])

scf_out = scf(dftcomm, crystal, kpts, grho, gwfn,
          numbnd_occ, is_spin, is_noncolin,
          rho_start=mag_start, occ_typ=occ,
          conv_thr=conv_thr, diago_thr_init=diago_thr_init,
          iter_printer=print_scf_status,
          ret_vxc=True)


print("SCF Routine has exited")
# print(qtmlogger)


# DFT: NSCF calculation for unshifted grid
# Observe that `maxiter` has been set to `1` and `diago_thr_init` has been set to a high value.


rho = scf_out[1].copy()
nscf_out = scf(dftcomm, crystal, kpts, grho, gwfn,
          numbnd_nscf, is_spin, is_noncolin,
          rho_start=rho, 
          occ_typ=occ,
          conv_thr=conv_thr, 
          diago_thr_init=(conv_thr/crystal.numel)/10,
          iter_printer=print_scf_status,
          maxiter=1,
          ret_vxc=True)

scf_converged_nscf, rho_nscf, l_wfn_kgrp, en_nscf, vxc = nscf_out


# #### DFT: NSCF calculation for shifted grid
# 
# Dielectric matrix calculation for the $q\to 0$ point will require energy eigenfunctions for a slightly shifted $k$-grid.


k_coords_q = kpts.k_cryst+np.array([[0,0,0.001]]).T
k_weights_q = np.ones(k_coords_q.shape[1])/k_coords_q.shape[1]
kpts_q = KList(recilat=crystal.recilat, k_coords=k_coords_q, k_weights=k_weights_q)

rho = scf_out[1].copy()
out_q = scf(dftcomm, crystal, kpts_q, grho, gwfn,
          numbnd_nscf, is_spin, is_noncolin,
          rho_start=rho, 
          occ_typ=occ,
          conv_thr=conv_thr, 
          diago_thr_init=(conv_thr/crystal.numel)/10,
          iter_printer=print_scf_status,
          maxiter=1)

scf_converged_nscf_q, rho_nscf_q, l_wfn_kgrp_q, en_nscf_q = out_q


print("Shifted NSCF Routine has exited")
# print(qtmlogger)


# Load Input Files
# Input data is handled by the ``EpsInp`` class.\
# The data can be provided either by constructing the ``EpsInp`` object or by reading BGW-compatible input file ``epsilon.inp``.\
# The attributes have been supplied with docstrings from BerkeleyGW's input specification, so they will be accessible directly in most IDEs.


from qtm.interfaces.bgw.epsinp import Epsinp

# Constructing input manually
# epsinp = Epsinp(epsilon_cutoff=1.2,
#                 use_wfn_hdf5=True,
#                 number_bands=8,
#                 write_vcoul=True,
#                 qpts=[[0.0,0.0,0.0]],
#                 is_q0=[True])

# Reading from epsilon.inp file
epsinp = Epsinp.from_epsilon_inp(filename=dirname+'epsilon.inp')
# print(epsinp)

# There is an analogous system to read SigmaInp
from qtm.interfaces.bgw.sigmainp import Sigmainp
sigmainp = Sigmainp.from_sigma_inp(filename=dirname+'sigma.inp')
# print(sigmainp)


# Initialize Epsilon Class
# 
# ``Epsilon`` class can be initialized by either directly passing the required `quantummasala.core` objects or by passing the input objects discussed earlier.


from qtm.gw.core import QPoints
from qtm.gw.epsilon import Epsilon
from qtm.klist import KList

kpts_gw =   KList(recilat=kpts.recilat,   cryst=kpts.k_cryst.T,   weights=kpts.k_weights)
kpts_gw_q = KList(recilat=kpts_q.recilat, cryst=kpts_q.k_cryst.T, weights=kpts_q.k_weights)

# Manual initialization
epsilon = Epsilon(
    crystal = crystal,
    gspace = grho,
    kpts = kpts_gw,
    kptsq = kpts_gw_q,        
    l_wfn = l_wfn_kgrp,
    l_wfnq = l_wfn_kgrp_q,
    qpts = QPoints.from_cryst(kpts.recilat, epsinp.is_q0, *epsinp.qpts),
    epsinp = epsinp,
)

# epsilon = Epsilon.from_data(wfndata=wfndata, wfnqdata=wfnqdata, epsinp=epsinp)


# The three main steps involved in the calculation have been mapped to the corresponding functions:
# 1.  ``matrix_elements``: Calculation of Planewave Matrix elements
# 2.  ``polarizability``: Calculation of RPA polarizability matrix $P$
# 3.  ``epsilon_inverse``: Calculation of (static) epsilon-inverse matrix


from tqdm import trange
from qtm.gw.core import reorder_2d_matrix_sorted_gvecs, sort_cryst_like_BGW


def calculate_epsilon(numq=None, writing=False):
    epsmats = []
    if numq is None:
        numq = epsilon.qpts.numq

    for i_q in trange(0, numq, desc="Epsilon> q-pt index"):
        # Create map between BGW's sorting order and QTm's sorting order
        gkspc = epsilon.l_gq[i_q]
        
        if i_q == epsilon.qpts.index_q0:
            key = gkspc.g_norm2
        else:
            key = gkspc.gk_norm2

        indices_gspace_sorted = sort_cryst_like_BGW(
            cryst=gkspc.g_cryst, key_array=key
        )

        # Calculate matrix elements
        M = next(epsilon.matrix_elements(i_q=i_q))

        # Calculate polarizability matrix (faster, but not memory-efficient)
        chimat = epsilon.polarizability(M)

        # Calculate polarizability matrix (memory-efficient)
        # chimat = epsilon.polarizability_active(i_q)
        
        # Calculate epsilon inverse matrix
        epsinv = epsilon.epsilon_inverse(i_q=i_q, polarizability_matrix=chimat, store=True)


        epsinv = reorder_2d_matrix_sorted_gvecs(epsinv, indices_gspace_sorted)
        epsilon.l_epsinv[i_q] = epsinv

        if i_q == epsilon.qpts.index_q0:
            if writing:
                epsilon.write_epsmat(
                    filename="test/epsilon/eps0mat_qtm.h5", epsinvmats=[epsinv]
                )
        else:
            epsmats.append(epsinv)
            
        if False:        
            # Compare the results with BGW's results
            if i_q == epsilon.qpts.index_q0:
                epsref = epsilon.read_epsmat(dirname + "eps0mat.h5")[0][0, 0]
            else:
                epsref = np.array(epsilon.read_epsmat(dirname + "epsmat.h5")[i_q - 1][0, 0])    

            # Calculate stddev between reference and calculated epsinv matrices
            std_eps = np.std(np.abs(epsref) - np.abs(epsinv)) / np.sqrt(np.prod(list(epsinv.shape)))
        
            epstol = 1e-16
            if np.abs(std_eps) > epstol:
                print(f"Standard deviation exceeded {epstol} tolerance: {std_eps}, for i_q:{i_q}")
                print(np.max(np.abs(epsinv-epsref)))
                indices = np.where(np.abs(epsinv)-np.abs(epsref)>1e-5)
                

    if writing:
        epsilon.write_epsmat(filename="test/epsilon/epsmat_qtm.h5", epsinvmats=epsmats)

calculate_epsilon()



# Sigma Calculation
# 
# Here we demonstate the calculation of diagonal matrix elements of $\Sigma_{\text{QP}}$. The input parameters for sigma calculation are being read from `sigma.inp` file, but the same parameters can also be provided by manually constructing a `SigmaInp` object. 
# 
# Here we will calculate $\bra{nk}\Sigma_{\text{QP}}\ket{nk}$ for the following k-points:
# - $\Gamma$: `k=(0,0,0)`
# - $L$: `k=(0.5,0.5,0)`
# - $X$: `k=(0,0.5,0)`


from qtm.gw.sigma import Sigma

outdir = dirname+"temp/"


sigma = Sigma.from_qtm_scf(
    crystal = crystal,
    gspace = grho,
    kpts = kpts_gw,
    kptsq=kpts_gw_q,
    l_wfn_kgrp=l_wfn_kgrp,
    l_wfn_kgrp_q=l_wfn_kgrp_q,
    sigmainp=sigmainp,
    epsinp = epsinp,
    epsilon=epsilon,
    rho=rho,
    vxc=vxc
)

# Alternatively, the Sigma object can also be intitialized from pw2bgw.x output data (after being procesed by wfn2hdf5.x).
# sigma = Sigma.from_data(
#     wfndata=wfndata,
#     wfnqdata=wfnqdata,
#     sigmainp=sigmainp,
#     epsinp=epsinp,
#     l_epsmats=epsilon.l_epsinv,
#     rho=rho,
#     vxc=vxc,
#     outdir=outdir,
# )


sigma_sx_cohsex_mat = sigma.sigma_sx_static(yielding=True)    
print("Sigma SX COHSEX")
sigma.pprint_sigma_mat(sigma_sx_cohsex_mat)


sigma_ch_cohsex_mat = sigma.sigma_ch_static()    
print("Sigma CH COHSEX")
sigma.pprint_sigma_mat(sigma_ch_cohsex_mat)


sigma.autosave=False
sigma.print_condition=True
cohsex_result = sigma.calculate_static_cohsex()


sigma.print_condition=True
print("Sigma CH COHSEX EXACT")
sigma_ch_exact_mat = sigma.sigma_ch_static_exact()    
sigma.pprint_sigma_mat(sigma_ch_exact_mat)


sigma.print_condition=False
sigma_sx_gpp = sigma.sigma_sx_gpp()    
print("Sigma SX GPP")
sigma.pprint_sigma_mat(sigma_sx_gpp)


sigma.print_condition=False
sigma_ch_gpp,_ = sigma.sigma_ch_gpp()    
print("Sigma CH GPP")
sigma.pprint_sigma_mat(sigma_ch_gpp)


gpp_result = sigma.calculate_gpp()


