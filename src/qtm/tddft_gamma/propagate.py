from typing import Optional, Callable

import numpy as np
from qtm.config import qtmconfig
from qtm.containers.field import FieldGType, FieldRType
from qtm.containers.wavefun import WavefunGType
from qtm.crystal.crystal import Crystal
from qtm.dft.kswfn import KSWfn
from qtm.dft.scf import EnergyData
from qtm.mpi.comm import QTMComm
from qtm.pot import ewald, hartree, xc
from qtm.pot.xc import check_libxc_func, get_libxc_func
from qtm.pseudo.loc import loc_generate_pot_rhocore
from qtm.pseudo.nloc import NonlocGenerator
from qtm.tddft_gamma.prop.etrs import normalize_rho
from tqdm import trange


def propagate(
    comm_world: QTMComm,
    crystal: Crystal,
    rho_start: FieldGType,
    wfn_gamma: list[list[KSWfn]],
    time_step: float,
    numstep: int,
    dipole_updater: Callable[[int, FieldGType, WavefunGType], None],
    libxc_func: Optional[tuple[str, str]] = None,
):
    # Begin setup ========================================
    config = qtmconfig

    numel = crystal.numel

    is_spin = len(wfn_gamma[0]) > 1
    is_noncolin = wfn_gamma[0][0].is_noncolin

    gspc_rho = rho_start.gspc
    gspc_wfn = gspc_rho
    gkspc = wfn_gamma[0][0].gkspc

    v_ion = rho_start.zeros(1)
    rho_core = rho_start.zeros(1)

    l_nloc = []
    for sp in crystal.l_atoms:
        v_ion_typ, rho_core_typ = loc_generate_pot_rhocore(sp, gspc_rho)
        v_ion += v_ion_typ
        rho_core += rho_core_typ
        l_nloc.append(NonlocGenerator(sp, gspc_wfn))
    v_ion = v_ion.to_r()

    rho_out: FieldGType
    v_hart: FieldRType
    v_xc: FieldRType
    v_loc: FieldRType

    en: EnergyData = EnergyData()

    if libxc_func is None:
        libxc_func = xc.get_libxc_func(crystal)
    else:
        xc.check_libxc_func(libxc_func)

    def compute_pot_local(rho_):
        nonlocal v_hart, v_xc, v_loc
        nonlocal rho_core
        v_hart, en.hartree = hartree.compute(rho_)
        v_xc, en.xc = xc.compute(rho_, rho_core, *libxc_func)
        v_loc = v_ion + v_hart + v_xc
        comm_world.bcast(v_loc._data)
        v_loc *= 1 / np.prod(gspc_wfn.grid_shape)
        return v_loc

    en.ewald = ewald.compute(crystal, gspc_rho)
    # End setup ========================================

    # Select expoper and propagator =====================
    prop_kwargs = {}
    print(f"config.tddft_exp_method: {config.tddft_exp_method}")
    if config.tddft_exp_method == "taylor":
        from .expoper.taylor import TaylorExp as PropOper

        prop_kwargs["order"] = config.taylor_order
    elif config.tddft_exp_method == "splitoper":
        from .expoper.splitoper import SplitOper as PropOper
    else:
        raise ValueError(
            "'config.tddft_exp_method' not recognized. "
            f"got {config.tddft_exp_method}."
        )

    if config.tddft_prop_method == "etrs":
        from .prop.etrs import prop_step
    elif config.tddft_prop_method == "splitoper":
        if config.tddft_exp_method != "splitoper":
            raise ValueError(
                "'config.tddft_exp_method' must be 'splitoper' when "
                "'config.tddft_prop_method' is set to 'splitoper. "
                f"got {config.tddft_exp_method} instead."
            )
        from .prop.splitoper import prop_step
    else:
        raise ValueError(
            "'config.tddft_prop_method' not recognized. "
            f"got {config.tddft_prop_method}."
        )
    # End selecting expoper and propagator ==============

    # Begin propagation ================================

    v_loc = compute_pot_local(rho_start)
    prop_gamma = PropOper(
        gkspc, is_spin, is_noncolin, v_loc, l_nloc, time_step, **prop_kwargs
    )

    rho = rho_start.copy()
    for istep in trange(numstep):
        # FIXME: We are implicitly assuming that the wavefunction is not spin-polarized.
        prop_step(wfn_gamma, rho, crystal.numel, compute_pot_local, prop_gamma)
        rho = (
            wfn_gamma[0][0].k_weight * wfn_gamma[0][0].compute_rho(ret_raw=True).to_g()
        )
        normalize_rho(rho, numel)
        dipole_updater(istep, rho)

    # End propagation ================================
